import json
from pathlib import Path
import sys
import subprocess
import os
import shutil
from zipfile import ZipFile

if sys.platform == "win32":
    try:
        import requests
    except Exception:
        print("installing requests...")
        python = sys.executable
        subprocess.check_call(
            f"{python} -m pip install requests",
            stdout=subprocess.DEVNULL,
        )
        import requests

PLATFORM = (
    "windows" if sys.platform == "win32" else "linux" if sys.platform == "linux" else ""
)


def check_version_and_platform() -> bool:
    version = sys.version_info
    return False if version.major != 3 and version.minor < 10 else PLATFORM != ""


def check_git_install() -> None:
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=PLATFORM == "linux",
        )
    except FileNotFoundError:
        print("ERROR: git is not installed, please install git")
        return False
    return True


# windows only
def set_execution_policy() -> None:
    try:
        subprocess.check_call(str(Path("installables/change_execution_policy.bat")))
    except subprocess.SubprocessError:
        try:
            subprocess.check_call(
                str(Path("installables/change_execution_policy_backup.bat"))
            )
        except subprocess.SubprocessError as e:
            print(f"Failed to change the execution policy with error:\n {e}")
            return False
    return True


def setup_accelerate(platform: str) -> None:
    if platform == "windows":
        path = Path(f"{os.environ['USERPROFILE']}")
    else:
        path = Path.home()
    path = path.joinpath(".cache/huggingface/accelerate/default_config.yaml")
    if path.exists():
        print("Default accelerate config already exists, skipping.")
        return
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open("default_config.yaml", "w") as f:
        f.write("command_file: null\n")
        f.write("commands: null\n")
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("deepspeed_config: {}\n")
        f.write("distributed_type: 'NO'\n")
        f.write("downcase_fp16: 'NO'\n")
        f.write("dynamo_backend: 'NO'\n")
        f.write("fsdp_config: {}\n")
        f.write("gpu_ids: '0'\n")
        f.write("machine_rank: 0\n")
        f.write("main_process_ip: null\n")
        f.write("main_process_port: null\n")
        f.write("main_training_function: main\n")
        f.write("megatron_lm_config: {}\n")
        f.write("mixed_precision: bf16\n")
        f.write("num_machines: 1\n")
        f.write("num_processes: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("tpu_name: null\n")
        f.write("tpu_zone: null\n")
        f.write("use_cpu: false")

    shutil.move("default_config.yaml", str(path.resolve()))


# windows only
def setup_cudnn():
    reply = None
    while reply not in ("y", "n"):
        reply = input(
            "Do you want to install the optional cudnn patch for faster "
            "training on high end 30X0 and 40X0 cards? (y/n): "
        ).casefold()
    if reply == "n":
        return

    r = requests.get(
        "https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip"
    )
    with open("cudnn.zip", "wb") as f:
        f.write(r.content)
    with ZipFile("cudnn.zip", "r") as f:
        f.extractall(path="cudnn_patch")
    shutil.move(
        str(Path("cudnn_patch/cudnn-windows-x86_64-8.6.0.163_cuda11-archive/bin")),
        "cudnn_windows",
    )
    os.mkdir("temp")
    r = requests.get(
        "https://raw.githubusercontent.com/bmaltais/kohya_ss/9c5bdd17499e3f677a5d7fa081ee0b4fccf5fd4a/tools/cudann_1.8_install.py"
    )
    with Path("temp/cudnn.py").open("wb") as f:
        f.write(r.content)
    subprocess.check_call(
        f"{Path('venv/Scripts/python.exe')} {Path('temp/cudnn.py')}".split(" ")
    )
    shutil.rmtree("temp")
    shutil.rmtree("cudnn_windows")
    shutil.rmtree("cudnn_patch")
    os.remove("cudnn.zip")


# windows only
def ask_10_series(venv_pip):
    reply = None
    while reply not in ("y", "n"):
        reply = input("Are you using a 10X0 series card? (y/n): ")
    if reply == "n":
        return False

    subprocess.check_call(
        f"{venv_pip} install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
    )
    subprocess.check_call(f"{venv_pip} install -r requirements.txt")
    subprocess.check_call(
        f"{venv_pip} install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"
    )
    subprocess.check_call(f"{venv_pip} install ../LyCORIS/.")
    subprocess.check_call(f"{venv_pip} install ../custom_scheduler/.")
    subprocess.check_call(f"{venv_pip} install bitsandbytes==0.35.0")
    subprocess.check_call(f"{venv_pip} install -r ../requirements.txt")

    shutil.copy(
        Path("../installables/libbitsandbytes_cudaall.dll"),
        Path("venv/Lib/site-packages/bitsandbytes"),
    )
    os.remove(Path("venv/Lib/site-packages/bitsandbytes/cuda_setup/main.py"))
    shutil.copy(
        Path("../installables/main.py"),
        Path("venv/Lib/site-packages/bitsandbytes/cuda_setup"),
    )
    return True


# windows only
def setup_windows(venv_pip):
    subprocess.check_call(
        f"{venv_pip} install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118"
    )
    subprocess.check_call(f"{venv_pip} install -r requirements.txt")
    subprocess.check_call(
        f"{venv_pip} install xformers --index-url https://download.pytorch.org/whl/cu118"
    )
    subprocess.check_call(f"{venv_pip} install ../LyCORIS/.")
    subprocess.check_call(f"{venv_pip} install ../custom_scheduler/.")
    subprocess.check_call(
        f"{venv_pip} install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl"
    )
    subprocess.check_call(f"{venv_pip} install -r ../requirements.txt")


# linux only
def setup_linux(venv_pip):
    subprocess.check_call(
        f"{venv_pip} install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118",
        shell=True,
    )
    subprocess.check_call(
        f"{venv_pip} install xformers --index-url https://download.pytorch.org/whl/cu118",
        shell=True,
    )
    subprocess.check_call(f"{venv_pip} install -r requirements.txt", shell=True)
    subprocess.check_call(f"{venv_pip} install ../LyCORIS/.", shell=True)
    subprocess.check_call(f"{venv_pip} install ../custom_scheduler/.", shell=True)
    subprocess.check_call(f"{venv_pip} install bitsandbytes scipy", shell=True)
    subprocess.check_call(f"{venv_pip} install -r ../requirements.txt", shell=True)


# colab only
def setup_colab(venv_pip):
    setup_linux(venv_pip)
    setup_accelerate("linux")


def ask_yes_no(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ")
    return reply == "y"


def setup_config(colab: bool = False) -> None:
    if colab:
        config = {
            "remote": True,
            "remote_mode": "cloudflared",
            "kill_tunnel_on_train_start": True,
            "kill_server_on_train_end": True,
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config, indent=2))
        return
    is_remote = ask_yes_no("are you using this remotely?")
    remote_mode = "none"
    if is_remote:
        remote_mode = (
            "ngrok" if ask_yes_no("do you want to use ngrok?") else "cloudflared"
        )
    ngrok_token = ""
    if remote_mode == "ngrok":
        ngrok_token = input(
            "copy paste your token from your ngrok dashboard (https://dashboard.ngrok.com/get-started/your-authtoken) (requires account): "
        )

    with open("config.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "remote": is_remote,
                    "remote_mode": remote_mode,
                    "ngrok_token": ngrok_token,
                },
                indent=2,
            )
        )


def main():
    if not check_version_and_platform() or not check_git_install():
        quit()

    subprocess.check_call("git submodule init", shell=PLATFORM == "linux")
    subprocess.check_call("git submodule update", shell=PLATFORM == "linux")

    if PLATFORM == "windows":
        print("setting execution policy to unrestricted")
        if not set_execution_policy():
            quit()

    setup_config(len(sys.argv) > 1 and sys.argv[1] == "colab")

    os.chdir("sd_scripts")
    if PLATFORM == "windows":
        pip = Path("venv/Scripts/pip.exe")
    else:
        pip = Path("venv/bin/pip")

    print("creating venv and installing requirements")
    subprocess.check_call(f"{sys.executable} -m venv venv", shell=PLATFORM == "linux")

    if len(sys.argv) > 1 and sys.argv[1] == "colab":
        setup_colab(pip)
        print("completed installing")
        quit()

    if PLATFORM == "windows":
        if not ask_10_series(pip):
            setup_windows(pip)
            setup_cudnn()
    else:
        setup_linux(pip)
    setup_accelerate(PLATFORM)

    print(
        "Completed installing, you can run the server via the run.bat or run.sh files"
    )


if __name__ == "__main__":
    main()
