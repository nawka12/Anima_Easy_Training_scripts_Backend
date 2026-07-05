from subprocess import check_call

from install_uv import ensure_uv, find_existing_venv
from installer import PLATFORM, setup_venv

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    # diffusion_pipe submodule + the two sub-submodules on its unconditional
    # import path: ComfyUI (`comfy`) and HunyuanVideo (`hyvideo`). Both are core
    # dependencies (see installer.py). Other sub-submodules are lazy-imported per
    # model type and are not pulled.
    check_call("git submodule update --init diffusion_pipe", shell=PLATFORM == "linux")
    check_call(
        "git -C diffusion_pipe submodule update --init "
        "submodules/ComfyUI submodules/HunyuanVideo",
        shell=PLATFORM == "linux",
    )

    uv = ensure_uv()
    # Backend-level venv, shared by the HTTP server and the deepspeed trainer.
    venv_path = find_existing_venv("venv") or "venv"
    setup_venv(uv, venv_path)


if __name__ == "__main__":
    main()
