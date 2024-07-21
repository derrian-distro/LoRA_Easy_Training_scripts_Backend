import json
from pathlib import Path
from pyngrok import ngrok
from utils.cloudflare_tunnel import TryCloudFlareConfig, Urls


class CloudflaredTunnel:
    def __init__(self) -> None:
        self.tunnel = TryCloudFlareConfig()
        self.running_tunnel: Urls = None

    def run_tunnel(self, port: int = 8000, config: Path | None = None) -> None:
        if config and config.is_file() and config.suffix == ".yml":
            self.running_tunnel = self.tunnel(config=config)
        else:
            self.running_tunnel = self.tunnel(port=port)

    def kill_service(self) -> bool:
        if self.running_tunnel:
            self.tunnel.terminate(self.running_tunnel.port)
            self.running_tunnel = None
        print("cloudflared process killed")
        return True


class NgrokTunnel:
    def __init__(self) -> None:
        config = Path("config.json")
        config_dict = json.loads(config.read_text()) if config.exists() else {}
        self.token = config_dict.get("ngrok_token", "")
        ngrok.set_auth_token(self.token)
        self.tunnel: ngrok.NgrokTunnel = None

    def run_tunnel(self, port: int = 8000) -> None:
        if self.tunnel:
            print(f"ngrok tunnel: {self.tunnel.public_url}")
            return
        try:
            self.tunnel = ngrok.connect(f"{port}")
            print(f"ngrok connected: {self.tunnel.public_url}")
        except Exception:
            print("ngrok ran into an issue, stopping ngrok process...")
            ngrok.kill()

    def kill_service(self) -> bool:
        if self.tunnel:
            ngrok.disconnect(self.tunnel.public_url)
        ngrok.kill()
        print("ngrok service killed")
        return True


def create_tunnel(config: dict) -> NgrokTunnel | CloudflaredTunnel:
    return (
        NgrokTunnel()
        if config.get("remote_mode", "cloudflared") == "ngrok"
        else CloudflaredTunnel()
    )
