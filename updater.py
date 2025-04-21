import os
from pathlib import Path

from installer import PLATFORM, setup_venv


def main():
    os.chdir("sd_scripts")

    if PLATFORM == "windows":
        pip = Path("venv/Scripts/pip.exe")
    else:
        pip = Path("venv/bin/pip")
    setup_venv(pip)


if __name__ == "__main__":
    main()
