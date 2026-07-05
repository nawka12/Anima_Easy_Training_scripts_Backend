import sys
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette import status
import json
from utils.validation import validate
from utils.process import write_configs
from pathlib import Path
import subprocess
import signal
import atexit
from utils.tunnel_service import CloudflaredTunnel, create_tunnel
import uvicorn
import os
from threading import Thread
import warnings
import logging

from transformers import CLIPTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
except Exception:
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not load CLIP tokenizer - offline and not cached. Continuing without it."
        )
        tokenizer = None


if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

if not Path("runtime_store").exists():
    Path("runtime_store").mkdir()


async def stop_server(_: Request = None) -> JSONResponse:
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
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000),
            config=Path(config_path) if config_path else None,
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
    return JSONResponse({"service_started": bool(app.state.TUNNEL)})


async def kill_tunnel_service(_: Request = None) -> JSONResponse:
    if not app.state.TUNNEL:
        return JSONResponse(
            {"killed": False, "reason": "No Tunnel Service Running"},
            status_code=400,
        )
    app.state.TUNNEL.kill_service()
    app.state.TUNNEL = None
    return JSONResponse({"killed": True, "reason": "Tunnel Service Successfully Killed"})


async def check_path(request: Request) -> JSONResponse:
    body = await request.body()
    body = json.loads(body)
    file_path = Path(body["path"])
    valid = False
    if body["type"] == "folder" and file_path.is_dir():
        valid = True
    if body["type"] == "file" and file_path.is_file() and file_path.suffix in body["extensions"]:
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
    passed, errors, main_cfg, dataset_cfg, sample_cfg, tags = validate(body)
    if not passed:
        return JSONResponse(errors, status_code=status.HTTP_400_BAD_REQUEST)
    main_path, dataset_path, sample_path = write_configs(main_cfg, dataset_cfg, sample_cfg)
    return JSONResponse(
        {
            "tags": tags,
            "main": main_path.read_text(encoding="utf-8"),
            "dataset": dataset_path.read_text(encoding="utf-8"),
            "sample": sample_path.read_text(encoding="utf-8") if sample_path else "",
        }
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
    if tokenizer is None:
        return JSONResponse(
            {"detail": "CLIP tokenizer is not available (offline and not cached)"},
            status_code=503,
        )
    text = request.query_params.get("text")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Original string:", text)
    # print("Tokenized string:", tokens)
    # print("Token IDs:", token_ids)
    return JSONResponse({"tokens": tokens, "token_ids": token_ids, "length": len(tokens)})


async def start_training(request: Request) -> JSONResponse:
    global server
    temp = json.loads(app.state.CONFIG.read_text())
    if "colab" in temp and temp["colab"]:
        await kill_tunnel_service()
        await stop_server()
        return
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )

    # GPU count + master port come from the (formerly "accelerate") settings.
    # Legacy sdxl/flux/anima/train_mode params are accepted and ignored.
    accelerate_enabled = request.query_params.get("accelerate_enabled", "False") == "True"
    num_gpus = (
        int(request.query_params.get("accelerate_num_processes", "1"))
        if accelerate_enabled
        else 1
    )
    master_port = int(request.query_params.get("accelerate_main_process_port", "29500"))
    resume = request.query_params.get("resume", "False").lower() in ("true", "1")

    server_config_dict = json.loads(app.state.CONFIG.read_text()) if app.state.CONFIG else {}
    main_config = Path("runtime_store/main.toml")
    if not main_config.is_file():
        return JSONResponse(
            {"detail": "No Previously Validated Args"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    train_script = Path("diffusion_pipe/train.py").resolve()
    # diffusion-pipe loads its bundled tokenizer/config dirs (configs/qwen3_06b,
    # configs/t5_old) via paths RELATIVE to the cwd, so the trainer must run with
    # its own directory as the working directory. Everything else in the command
    # (train.py, --config, and the paths inside the TOMLs) is already absolute.
    dp_dir = train_script.parent
    cmd = [
        "deepspeed",
        f"--num_gpus={num_gpus}",
        f"--master_port={master_port}",
        str(train_script),
        "--config",
        str(main_config.resolve()),
    ]
    if resume:
        cmd.append("--resume_from_checkpoint")
    print(f"Launching diffusion-pipe: {' '.join(cmd)}")

    env = {**os.environ, "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}
    # Reduce CUDA fragmentation OOMs (large "reserved but unallocated" gaps) by
    # letting the allocator grow segments. setdefault so an explicit user
    # override in the environment still wins.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # deepspeed forks worker processes; start them in a fresh process group so
    # stop_training can signal the whole group (terminate() on the launcher
    # alone orphans the workers, which keep the GPU memory pinned).
    app.state.TRAINING_THREAD = subprocess.Popen(
        cmd, env=env, cwd=str(dp_dir), preexec_fn=os.setsid
    )
    if (
        "kill_tunnel_on_train_start" in server_config_dict
        and server_config_dict["kill_tunnel_on_train_start"]
    ):
        app.state.TUNNEL.kill_service()
        app.state.TUNNEL = None
    if "kill_server_on_train_end" in server_config_dict and server_config_dict["kill_server_on_train_end"]:
        app.state.MONITOR_THREAD = Thread(target=monitor_training_thread, daemon=True)
        app.state.MONITOR_THREAD.start()
    return JSONResponse({"detail": "Training Started", "training": True})


def _kill_training_group(sig: int) -> bool:
    """Signal the whole trainer process group; return True if a live trainer
    was signalled.

    deepspeed is launched in its own process group (os.setsid), so signalling
    the group kills the forked workers too and releases the GPU + master port.
    Falls back to the launcher pid if the group is already gone.
    """
    thread = app.state.TRAINING_THREAD
    if not thread or thread.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(thread.pid), sig)
    except ProcessLookupError:
        try:
            thread.send_signal(sig)
        except ProcessLookupError:
            return False
    return True


async def stop_training(request: Request) -> JSONResponse:
    force = request.query_params.get("force", "False").lower() in ("true", "1")
    if not _kill_training_group(signal.SIGKILL if force else signal.SIGTERM):
        return JSONResponse(
            {"detail": "Not Currently Training"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if force:
        return JSONResponse({"detail": "Training Thread Killed"})
    return JSONResponse({"detail": "Training Thread Requested to Die"})


async def start_resize(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "Training Already Running"}, status_code=status.HTTP_409_CONFLICT)
    data = await request.body()
    data: list[str] = json.loads(data)
    python = sys.executable
    app.state.TRAINING_THREAD = subprocess.Popen([python, f"{Path('utils/resize_lora.py').resolve()}"] + data)
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
app.state.TRAINING_THREAD = None
app.state.CONFIG = Path("config.json")
app.state.MONITOR_THREAD = None

if not app.state.CONFIG.exists():
    with app.state.CONFIG.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"remote": False, "port": 8000}, indent=2))

config_data = json.loads(app.state.CONFIG.read_text())
if config_data.get("remote", False):
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = config_data.get("cloudflared_config_path", None)
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000), config=Path(config_path) if config_path else None
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
uvi_config = uvicorn.Config(
    app,
    host=config_data.get("host", "0.0.0.0"),
    loop="asyncio",
    log_level="critical",
    port=config_data.get("port", 8000),
)
server = uvicorn.Server(config=uvi_config)


def _cleanup_training_on_exit():
    """Ensure the detached deepspeed trainer doesn't outlive the server.

    The trainer runs in its own process group, so a Ctrl-C on run.sh (SIGINT)
    or a `kill` (SIGTERM) reaches uvicorn but NOT the trainer group, orphaning
    it — it keeps holding the GPU and the master port (EADDRINUSE on the next
    run). uvicorn handles SIGINT/SIGTERM gracefully and returns from run(), so
    this atexit hook fires on the way out and tears the group down.
    """
    if _kill_training_group(signal.SIGTERM):
        try:
            app.state.TRAINING_THREAD.wait(timeout=10)
        except Exception:
            _kill_training_group(signal.SIGKILL)


atexit.register(_cleanup_training_on_exit)

if __name__ == "__main__":
    server.run()
