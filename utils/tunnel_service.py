from sys import platform
from pathlib import Path
from multiprocessing import Process
import subprocess

import requests


class Tunnel:
    def __init__(self, download_folder: Path = Path("runtime_store")) -> None:
        self.folder = download_folder
        self.executable = self.folder.joinpath(
            "cloudflared-linux-amd64"
            if platform == "linux"
            else "cloudflared-windows-amd64.exe"
        )
        self.process = Process(target=self.run_tunnel)
        if not self.folder.exists():
            self.folder.mkdir()
        if not self.executable.exists():
            urls = [
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
            ]
            response = requests.get(urls[0] if platform == "linux" else urls[1])
            with self.executable.open("wb") as f:
                f.write(response.content)
            if platform == "linux":
                self.executable.chmod(711)
        self.process.start()

    def run_tunnel(self):
        print("running tunnel")
        subprocess.check_call(
            f"{'./' if platform == 'linux' else ''}{self.executable} tunnel --url http://127.0.0.1:8000",
            shell=platform == "linux",
        )
