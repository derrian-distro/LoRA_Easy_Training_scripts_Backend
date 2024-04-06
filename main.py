import sys
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette import status
import json
from utils.validation import validate
from utils.process import process_args, process_dataset_args
from pathlib import Path
import subprocess
from utils.tunnel_service import CloudflaredTunnel, create_tunnel
import uvicorn
import os
from threading import Thread

from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

if not Path("runtime_store").exists():
    Path("runtime_store").mkdir()


async def stop_server(_: Request) -> JSONResponse:
    global server
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "training still running"})
    server.should_exit = True
    server.force_exit = True


async def start_tunnel_service(request: Request) -> JSONResponse:
    config_data = json.loads(app.state.CONFIG.read_text())
    if app.state.TUNNEL:
        return JSONResponse({"service_started": False}, status_code=409)
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = request.query_params.get(
            "config_path", config_data.get("cloudflared_config_path", None)
        )
        if config_path:
            config_path = Path(config_path)
        app.state.TUNNEL.run_tunnel(config=Path(config_path) if config_path else None)
    else:
        app.state.TUNNEL.run_tunnel()
    return JSONResponse({"service_started": bool(app.state.TUNNEL)})


async def kill_tunnel_service(_: Request) -> JSONResponse:
    if not app.state.TUNNEL:
        return JSONResponse(
            {"killed": False, "reason": "No Tunnel Service Running"},
            status_code=400,
        )
    app.state.TUNNEL.kill_service()
    app.state.TUNNEL = None
    return JSONResponse(
        {"killed": True, "reason": "Tunnel Service Successfully Killed"}
    )


async def check_path(request: Request) -> JSONResponse:
    body = await request.body()
    body = json.loads(body)
    file_path = Path(body["path"])
    valid = False
    if body["type"] == "folder" and file_path.is_dir():
        valid = True
    if (
        body["type"] == "file"
        and file_path.is_file()
        and file_path.suffix in body["extensions"]
    ):
        valid = True
    return JSONResponse({"valid": valid})


async def validate_inputs(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    body = await request.body()
    body = json.loads(body)
    passed_validation, sdxl, errors, args, dataset_args, tags = validate(body)
    if passed_validation:
        output_args, _ = process_args(args)
        output_dataset_args, _ = process_dataset_args(dataset_args)
        final_args = {"args": output_args, "dataset": output_dataset_args, "tags": tags}
        return JSONResponse(final_args)
    return JSONResponse(
        errors,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def is_training(_: Request) -> JSONResponse:
    exit_id = app.state.TRAINING_THREAD.poll() if app.state.TRAINING_THREAD else 0
    return JSONResponse(
        {
            "training": exit_id is None,
            "errored": exit_id is not None and exit_id != 0,
        }
    )

async def tokenize_text(request: Request) -> JSONResponse:
    text = request.query_params.get("text")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Original string:", text)
    # print("Tokenized string:", tokens)
    # print("Token IDs:", token_ids)
    return JSONResponse({"tokens": tokens, "token_ids": token_ids, "length": len(tokens)})


async def start_training(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    is_sdxl = request.query_params.get("sdxl", "False") == "True"
    train_type = request.query_params.get("train_mode", "lora")
    match [train_type, is_sdxl]:
        case ['lora', False]:
            app.state.TRAIN_SCRIPT = "train_network.py"
        case ['lora', True]:
            app.state.TRAIN_SCRIPT = "sdxl_train_network.py"
        case ['textual_inversion', False]:
            app.state.TRAIN_SCRIPT = "train_textual_inversion.py"
        case ['textual_inversion', True]:
            app.state.TRAIN_SCRIPT = "sdxl_train_textual_inversion.py"
        case _:
            print("Unknown training request: {request.query_params}")
            return JSONResponse({
                        "detail": "Invalid Train Parameters",
                        "sdxl": is_sdxl,
                        "train_type": train_type
                    },
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    server_config_dict = (
        json.loads(app.state.CONFIG.read_text()) if app.state.CONFIG else {}
    )
    python = sys.executable
    config = Path("runtime_store/config.toml")
    dataset = Path("runtime_store/dataset.toml")
    if not config.is_file() or not dataset.is_file():
        return JSONResponse(
            {"detail": "No Previously Validated Args"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    print(app.state.TRAIN_SCRIPT)
    app.state.TRAINING_THREAD = subprocess.Popen(
        [
            f"{python}",
            f"{Path(f'sd_scripts/{app.state.TRAIN_SCRIPT}').resolve()}",
            f"--config_file={config.resolve()}",
            f"--dataset_config={dataset.resolve()}",
        ]
    )
    if (
        "kill_tunnel_on_train_start" in server_config_dict
        and server_config_dict["kill_tunnel_on_train_start"]
    ):
        app.state.TUNNEL.kill_service()
        app.state.TUNNEL = None
    if (
        "kill_server_on_train_end" in server_config_dict
        and server_config_dict["kill_server_on_train_end"]
    ):
        app.state.MONITOR_THREAD = Thread(target=monitor_training_thread, daemon=True)
        app.state.MONITOR_THREAD.start()
    return JSONResponse({"detail": "Training Started", "training": True})


async def stop_training(request: Request) -> JSONResponse:
    force = bool(request.query_params.get("force", False))
    if not app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is not None:
        return JSONResponse(
            {"detail": "Not Currently Training"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if force:
        app.state.TRAINING_THREAD.stderr = None
        app.state.TRAINING_THREAD.kill()
        return JSONResponse({"detail": "Training Thread Killed"})
    else:
        app.state.TRAINING_THREAD.terminate()
    return JSONResponse({"detail": "Training Thread Requested to Die"})


async def start_resize(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"}, status_code=status.HTTP_409_CONFLICT
        )
    data = await request.body()
    data: list[str] = json.loads(data)
    python = sys.executable
    app.state.TRAINING_THREAD = subprocess.Popen(
        [python, f"{Path('utils/resize_lora.py').resolve()}"] + data
    )
    return JSONResponse({"detail": "Resizing Started"})


def monitor_training_thread():
    if not app.state.TRAINING_THREAD:
        return
    global server
    app.state.TRAINING_THREAD.wait()
    server.should_exit = True
    server.force_exit = True


routes = [
    Route("/stop_server", stop_server, methods=["GET"]),
    Route("/start_tunnel_service", start_tunnel_service, methods=["GET"]),
    Route("/kill_tunnel_service", kill_tunnel_service, methods=["GET"]),
    Route("/check_path", check_path, methods=["POST"]),
    Route("/validate", validate_inputs, methods=["POST"]),
    Route("/is_training", is_training, methods=["GET"]),
    Route("/train", start_training, methods=["GET"]),
    Route("/tokenize", tokenize_text, methods=["GET"]),
    Route("/stop_training", stop_training, methods=["GET"]),
    Route("/resize", start_resize, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)
app.state.TRAIN_SCRIPT = None
app.state.TRAINING_THREAD = None
app.state.CONFIG = Path("config.json")
app.state.MONITOR_THREAD = None

if not app.state.CONFIG.exists():
    with app.state.CONFIG.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"remote": False}, indent=2))

config_data = json.loads(app.state.CONFIG.read_text())
if config_data.get("remote", False):
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = config_data.get("cloudflared_config_path", None)
        app.state.TUNNEL.run_tunnel(config=Path(config_path) if config_path else None)
    else:
        app.state.TUNNEL.run_tunnel()
uvi_config = uvicorn.Config(app, host="0.0.0.0", loop="asyncio", log_level="critical")
server = uvicorn.Server(config=uvi_config)

if __name__ == "__main__":
    server.run()
