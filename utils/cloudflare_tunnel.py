import atexit
from pathlib import Path
import re
import subprocess
from typing import NamedTuple
from pycloudflared.try_cloudflare import TryCloudflare
from pycloudflared.util import download, get_info
import yaml


class Urls(NamedTuple):
    tunnel: str
    metrics: str
    process: subprocess.Popen
    port: int


class TryCloudFlareConfig(TryCloudflare):
    def __init__(self):
        super().__init__()
        self.url_pattern = re.compile(r"(?P<url>https?://\S+\.trycloudflare\.com)")
        self.metrics_pattern = re.compile(r"(?P<url>127\.0\.0\.1:\d+/metrics)")

    def __call__(
        self,
        port: int | str = 8000,
        metrics_port: int | str | None = None,
        verbose: bool = True,
        config: Path | None = None,
    ) -> Urls:
        info = get_info()
        if not Path(info.executable).exists():
            download(info)

        failed_config = False
        if config and config.is_file() and config.suffix == ".yml":
            temp = yaml.safe_load(config.open("r", encoding="utf-8"))
            if "url" not in temp:
                print(
                    "url not found in config, defaulting to url http://127.0.0.1:8000"
                )
                failed_config = True
            port = temp.get("url", ":8000").split(":")[1]

        port = int(port)
        if port in self.running:
            urls = self.running[port]
            if verbose:
                self._print(urls.tunnel, urls.metrics)
            return urls

        if (
            config
            and config.is_file()
            and config.suffix == ".yml"
            and not failed_config
        ):
            args = [info.executable, "tunnel", "--config", config.as_posix()]
        else:
            args = [info.executable, "tunnel", "--url", f"http://127.0.0.1:{port}"]
            if metrics_port is not None:
                args += [
                    "--metrics",
                    f"127.0.0.1:{metrics_port}",
                ]

        if info.system == "darwin" and info.machine == "arm64":
            args = ["arch", "-x86_64"] + args

        cloudflared = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )

        atexit.register(cloudflared.terminate)

        tunnel_url = metrics_url = ""

        lines = 20
        for _ in range(lines):
            line = cloudflared.stderr.readline()

            url_match = self.url_pattern.search(line)
            metric_match = self.metrics_pattern.search(line)
            if url_match:
                tunnel_url = url_match.group("url")
            if metric_match:
                metrics_url = "http://" + metric_match.group("url")

            if tunnel_url and metrics_url:
                break

        else:
            raise RuntimeError("Cloudflared failed to start")

        urls = Urls(tunnel_url, metrics_url, cloudflared, port)
        if verbose:
            self._print(urls.tunnel, urls.metrics)

        self.running[port] = urls
        return urls


try_cloudflare = TryCloudFlareConfig()
