from pathlib import Path
from subprocess import check_call
import os
from installer import PLATFORM, setup_windows, setup_linux


def main():
    check_call("git submodule init", shell=PLATFORM == "linux")
    check_call("git submodule update", shell=PLATFORM == "linux")
    os.chdir("sd_scripts")
    if PLATFORM == "windows":
        setup_windows(Path("venv/Scripts/pip.exe"))
    else:
        setup_linux(Path("venv/bin/pip"))


if __name__ == "__main__":
    main()
