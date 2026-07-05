from subprocess import check_call

from install_uv import ensure_uv, find_existing_venv
from installer import PLATFORM, setup_venv

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    # Only the diffusion_pipe submodule is needed (see installer.py); its own
    # heavy sub-submodules are for other model families and are not pulled.
    check_call("git submodule update --init diffusion_pipe", shell=PLATFORM == "linux")

    uv = ensure_uv()
    # Backend-level venv, shared by the HTTP server and the deepspeed trainer.
    venv_path = find_existing_venv("venv") or "venv"
    setup_venv(uv, venv_path)


if __name__ == "__main__":
    main()
