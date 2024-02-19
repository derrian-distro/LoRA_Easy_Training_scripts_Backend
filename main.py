import contextlib
import sys
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette import status
import json
from utils.validation import validate
from utils.process import process_args, process_dataset_args
from pathlib import Path
import subprocess
import threading
from utils.tunnel_service import Tunnel
import uvicorn

if not Path("runtime_store").exists():
    Path("runtime_store").mkdir()


async def validate_inputs(request: Request) -> Response:
    if app.state.TRAINING:
        return Response(
            json.dumps({"detail": "training already running"}),
            status_code=status.HTTP_409_CONFLICT,
        )
    body = await request.body()
    body = json.loads(body)
    passed_validation, sdxl, errors, args, dataset_args = validate(body)
    if passed_validation:
        app.state.SD_TYPE = "sdxl_train_network.py" if sdxl else "train_network.py"
        output_args, _ = process_args(args)
        output_dataset_args, _ = process_dataset_args(dataset_args)
        final_args = {"args": output_args, "dataset": output_dataset_args}
        return Response(json.dumps(final_args, indent=2))
    return Response(
        json.dumps(errors),
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def train_request(_: Request) -> Response:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.is_alive():
        return Response(
            json.dumps({"detail": "training already running"}),
            status_code=status.HTTP_409_CONFLICT,
        )
    args = Path("runtime_store/config.toml")
    dataset = Path("runtime_store/dataset.toml")
    if not args.exists() or not dataset.exists():
        return Response(
            json.dumps({"detail": "No Previously Validated Args"}),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    app.state.TRAINING_THREAD = threading.Thread(target=train, daemon=True)
    app.state.TRAINING_THREAD.start()
    return Response(json.dumps({"detail": "training started successfully"}))


async def is_training(_: Request) -> Response:
    return Response(
        json.dumps({"training": app.state.TRAINING, "errored": app.state.ERROR})
    )


async def stop_server(_: Request) -> Response:
    global server
    if app.state.TRAINING:
        return Response(json.dumps({"detail": "training still running"}))
    server.should_exit = True
    server.force_exit = True


def train() -> None:
    python = sys.executable
    app.state.TRAINING = True
    app.state.ERROR = False
    try:
        subprocess.check_call(
            f"{python} {Path(f'sd_scripts/{app.state.SD_TYPE}').resolve()} --config_file={Path('runtime_store/config.toml').resolve()} --dataset_config={Path('runtime_store/dataset.toml').resolve()}",
            shell=sys.platform == "linux",
        )
    except subprocess.SubprocessError:
        app.state.TRAINING = False
        app.state.ERROR = True
        return
    app.state.TRAINING = False
    app.state.ERROR = False


def startup():
    with contextlib.suppress(AttributeError):
        app.state.TUNNEL.process.start()


routes = [
    Route("/validate", validate_inputs, methods=["POST"]),
    Route("/train", train_request, methods=["GET"]),
    Route("/is_training", is_training, methods=["GET"]),
    Route("/stop_server", stop_server, methods=["GET"]),
]

app = Starlette(debug=True, routes=routes, on_startup=[startup])
app.state.SD_TYPE = "train_network.py"
app.state.TRAINING_THREAD = None
app.state.TRAINING = False
app.state.ERROR = False
app.state.CONFIG = Path("config.json")

if not app.state.CONFIG.exists():
    with app.state.CONFIG.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"remote": False}, indent=2))

config_data = json.loads(app.state.CONFIG.read_text())
if "remote" in config_data and config_data["remote"]:
    app.state.TUNNEL = Tunnel()

uvi_config = uvicorn.Config(app, loop="asyncio")
server = uvicorn.Server(config=uvi_config)

if __name__ == "__main__":
    server.run()
