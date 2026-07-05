import json
from pathlib import Path
import sys
import subprocess
import os
import shutil

from install_uv import ensure_uv, create_venv, uv_pip_install, get_venv_python_version

PLATFORM = "windows" if sys.platform == "win32" else "linux" if sys.platform == "linux" else ""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_git_install() -> bool:
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=PLATFORM == "linux",
        )
    except FileNotFoundError:
        logger.error("ERROR: git is not installed, please install git")
        return False
    return True

def setup_accelerate(platform: str) -> None:
    if platform == "windows":
        path = Path(f"{os.environ['USERPROFILE']}")
    else:
        path = Path.home()
    path = path.joinpath(".cache/huggingface/accelerate/default_config.yaml")
    if path.exists():
        logger.info("Default accelerate config already exists, skipping.")
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


# flash-attention wheel URLs for each platform/python combination
FLASH_ATTN_WHEELS = {
    ("win32", 11): "https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.0.post2/flash_attn-2.8.0.post2+cu128torch2.7.1cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl",
    ("linux", 10): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    ("linux", 11): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    ("linux", 12): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    ("linux", 13): "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp313-cp313-linux_x86_64.whl",
}


def _install_flash_attn(uv: str, venv_path: str = "venv") -> None:
    """Install flash-attention wheel for the current platform and Python version.

    Detects the Python version from the venv executable. This is done
    explicitly before requirements.txt to avoid uv reinstalling flash-attn
    due to version normalization differences (FALSE vs false in local
    version identifiers).
    """
    python_version = get_venv_python_version(venv_path)
    python_minor = int(python_version.split(".")[1])
    platform_key = "win32" if sys.platform == "win32" else "linux"
    wheel_url = FLASH_ATTN_WHEELS.get((platform_key, python_minor))

    if wheel_url:
        logger.info(f"Installing flash-attn for {platform_key} Python 3.{python_minor}...")
        uv_pip_install(uv, "--no-deps", wheel_url, venv_path=venv_path)
    else:
        logger.info(f"No flash-attn wheel for {platform_key} Python 3.{python_minor}, skipping")


def setup_venv(uv: str, venv_path: str = "venv"):
    """Install diffusion-pipe + backend + frontend packages into the venv.

    NOTE: the torch / flash-attn / CUDA pins below need validation on real GPU
    hardware during the Phase 0 smoke test (BACK-MIGRATE.md §5). They are kept
    consistent with the flash-attn wheels declared above (torch 2.7 / cu128).
    """
    uv_pip_install(uv, "-U", "typing-extensions==4.15.0", venv_path=venv_path)

    # torch must be installed before deepspeed (which builds against it).
    uv_pip_install(
        uv,
        "-U", "torch~=2.7.1", "torchvision~=0.22.1", "numpy~=2.2.6",
        "--index-url", "https://download.pytorch.org/whl/cu128",
        venv_path=venv_path,
    )

    # Pre-install flash-attn explicitly to avoid uv version normalization issues
    # (uv normalizes FALSE -> false in local version identifiers, causing reinstall loops)
    _install_flash_attn(uv, venv_path)

    # diffusion-pipe's own requirements (includes deepspeed, transformers,
    # diffusers, peft, torch-optimi, pytorch-optimizer, wandb, tensorboard, ...).
    uv_pip_install(uv, "-r", "diffusion_pipe/requirements.txt", venv_path=venv_path)
    # Starlette / uvicorn / tunnel deps for the HTTP server itself.
    uv_pip_install(uv, "-r", "requirements.txt", venv_path=venv_path)
    # NOTE: the backend is self-contained (it can be cloned standalone on a
    # remote/GPU box). It must NOT depend on the frontend's UI requirements
    # (PySide6 / qt-material) — those install with the frontend, not here.


# colab only
def setup_colab(uv: str, venv_path: str = "venv"):
    setup_venv(uv, venv_path)
    setup_accelerate("linux")


def ask_yes_no(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ")
    return reply == "y"


def setup_config(colab: bool = False, local: bool = False) -> None:
    if colab:
        config = {
            "remote": True,
            "remote_mode": "cloudflared",
            "kill_tunnel_on_train_start": True,
            "kill_server_on_train_end": True,
            "colab": True,
            "port": 8000,
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config, indent=2))
        return
    is_remote = False if local else ask_yes_no("are you using this remotely?")
    remote_mode = "none"
    if is_remote:
        remote_mode = "ngrok" if ask_yes_no("do you want to use ngrok?") else "cloudflared"
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
                    "port": 8000,
                },
                indent=2,
            )
        )


def main():
    if not check_git_install():
        quit()

    # Pull the diffusion_pipe submodule, plus the two sub-submodules on its
    # unconditional import path: ComfyUI (utils/dataset.py -> comfy, and
    # models/base.py -> comfy.utils/sd/sd1_clip) and HunyuanVideo
    # (utils/patches.py -> hyvideo.text_encoder). train.py loads both of these
    # modules for EVERY run, so both submodules are required for all training,
    # Anima included. The remaining sub-submodules (Cosmos, Lumina, HiDream,
    # ...) are imported lazily only when their model type is selected
    # (train.py dispatches on model.type), so we don't pull them.
    subprocess.check_call(
        "git submodule update --init diffusion_pipe", shell=PLATFORM == "linux"
    )
    subprocess.check_call(
        "git -C diffusion_pipe submodule update --init "
        "submodules/ComfyUI submodules/HunyuanVideo",
        shell=PLATFORM == "linux",
    )

    setup_config(
        len(sys.argv) > 1 and sys.argv[1] == "colab",
        len(sys.argv) > 1 and sys.argv[1] == "local",
    )

    logger.info("creating venv and installing requirements")
    uv = ensure_uv()
    # Backend-level venv (shared by the HTTP server and the deepspeed trainer).
    # Python 3.12 is required: diffusion-pipe's latent cache uses
    # sqlite3.connect(autocommit=...), a 3.12+ API (utils/cache.py), and
    # upstream itself targets 3.12.
    venv_path = create_venv(uv, "venv", "3.12")

    if len(sys.argv) > 1 and sys.argv[1] == "colab":
        setup_colab(uv, venv_path)
        logger.info("completed installing")
        quit()

    setup_venv(uv, venv_path)

    logger.info("Completed installing, you can run the server via the run.bat or run.sh files")


if __name__ == "__main__":
    main()
