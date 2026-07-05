"""Validate the frontend payload and translate it into diffusion-pipe configs.

The frontend still POSTs the same envelope to ``/validate``::

    {"args": {...groups...}, "dataset": {...}, "accelerate": {...}}

but the *vocabulary* inside each group is now the diffusion-pipe one (see
BACK-MIGRATE.md §4 / §9). ``validate`` returns three ready-to-serialize dicts
-- one per output TOML -- plus caption tag counts:

    (passed, errors, main_config, dataset_config, sample_config, tags)

``sample_config`` is ``None`` when there are no sample prompts. Downstream,
``utils.process.write_configs`` writes ``runtime_store/{main,dataset,sample}.toml``
and cross-references them by absolute path.
"""
from __future__ import annotations

from pathlib import Path
import math

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}

# sd_scripts mixed_precision names -> diffusion-pipe dtype names.
DTYPE_MAP = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "float": "float32",
    "no": "float32",
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float8": "float8",
    "fp8": "float8",
}

# Optimizer name map. Anything not here is passed through verbatim so the
# pytorch_optimizer library can resolve it (train.py getattr, case-sensitive).
# Keys are matched case-insensitively.
OPTIMIZER_NAME_MAP = {
    "adamw": "adamw_optimi",
    "adamw_optimi": "adamw_optimi",
    "adamwoptimi": "adamw_optimi",
    "adamw8bit": "AdamW8bitKahan",
    "adamw8bitkahan": "AdamW8bitKahan",
    "stableadamw": "stableadamw",
    "sgd": "sgd",
    "offload": "offload",
    "automagic": "automagic",
    "prodigy": "Prodigy",
    "came": "CAME",
}

VALID_TIMESTEP_METHODS = {"logit_normal", "uniform"}
VALID_LR_SCHEDULERS = {"constant", "linear", "cosine"}
ADAPTER_TYPES = {"lora", "lokr"}


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _as_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if value is None:
        return default
    return bool(value)


def _as_float(value, default=None):
    if _blank(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=None):
    if _blank(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first(group: dict, *keys, default=None):
    """Return the first present, non-blank value among ``keys``."""
    for key in keys:
        if key in group and not _blank(group[key]):
            return group[key]
    return default


def _toml_literal(text: str):
    """Best-effort parse of a free-form extra_arg value into a TOML scalar."""
    s = text.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return text


def validate(body: dict):
    errors: list[str] = []
    if "args" not in body:
        errors.append("'args' is not present in the payload")
    if "dataset" not in body:
        errors.append("'dataset' is not present in the payload")
    if errors:
        return False, errors, {}, {}, None, {}

    args = body.get("args", {}) or {}
    dataset = body.get("dataset", {}) or {}
    accelerate = body.get("accelerate", {}) or {}

    general = args.get("general_args", {}) or {}
    model_group = args.get("anima_args", args.get("model_args", {})) or {}
    network = args.get("network_args", {}) or {}
    optimizer = args.get("optimizer_args", {}) or {}
    saving = args.get("saving_args", {}) or {}
    sample = args.get("sample_args", {}) or {}
    logging_group = args.get("logging_args", {}) or {}
    extra = args.get("extra_args", {}) or {}

    ds_general = dataset.get("general_args", {}) or {}
    bucket = dataset.get("bucket_args", {}) or {}
    caption_args = dataset.get("caption_args", {}) or {}
    subsets_in = dataset.get("subsets", []) or []

    num_gpus = (
        _as_int(accelerate.get("num_processes"), 1)
        if _as_bool(accelerate.get("enabled"))
        else 1
    ) or 1

    main: dict = {}

    # ---- dtype / precision -------------------------------------------------
    mixed_precision = _first(general, "mixed_precision", "dtype", default="bf16")
    model_dtype = DTYPE_MAP.get(str(mixed_precision).lower(), "bfloat16")

    # ---- [model] -----------------------------------------------------------
    model_cfg, model_errors = _build_model(model_group, model_dtype)
    errors += model_errors

    # ---- [adapter] (omitted for full fine-tune) ----------------------------
    adapter_cfg, adapter_errors = _build_adapter(network, model_dtype)
    errors += adapter_errors

    # ---- [optimizer] + top-level LR/loss/scheduler keys --------------------
    optimizer_cfg, opt_errors = _build_optimizer(optimizer, main)
    errors += opt_errors

    # ---- general / training top-level --------------------------------------
    main["epochs"] = _as_int(_first(general, "epochs", "max_train_epochs"), 1)
    grad_acc = _as_int(_first(general, "gradient_accumulation_steps"), 1) or 1
    main["gradient_accumulation_steps"] = grad_acc
    main["micro_batch_size_per_gpu"] = _as_int(
        _first(ds_general, "micro_batch_size_per_gpu", "batch_size"), 1
    ) or 1
    main["pipeline_stages"] = _as_int(_first(general, "pipeline_stages"), 1) or 1
    main["activation_checkpointing"] = _as_bool(
        _first(general, "activation_checkpointing", "gradient_checkpointing", default=True),
        default=True,
    )
    if _as_bool(general.get("reentrant_activation_checkpointing")):
        main["reentrant_activation_checkpointing"] = True
    main["partition_method"] = _first(general, "partition_method", default="parameters")
    blocks_to_swap = _as_int(general.get("blocks_to_swap"))
    if blocks_to_swap:
        main["blocks_to_swap"] = blocks_to_swap
    if _as_bool(general.get("compile")):
        main["compile"] = True
    main["steps_per_print"] = _as_int(_first(general, "steps_per_print"), 1) or 1
    main["caching_batch_size"] = _as_int(_first(general, "caching_batch_size"), 1) or 1
    map_num_proc = _as_int(_first(general, "map_num_proc", "max_data_loader_n_workers"))
    if map_num_proc:
        main["map_num_proc"] = map_num_proc

    # pipeline_stages must divide the GPU count sensibly.
    if num_gpus % main["pipeline_stages"] != 0:
        errors.append(
            f"pipeline_stages ({main['pipeline_stages']}) must divide the GPU "
            f"count ({num_gpus})."
        )
    if blocks_to_swap and main["pipeline_stages"] != 1:
        errors.append("blocks_to_swap requires pipeline_stages = 1.")
    if blocks_to_swap and adapter_cfg is None:
        errors.append("blocks_to_swap requires an adapter (LoRA/LoKr), not full fine-tune.")

    # ---- saving ------------------------------------------------------------
    saving_errors = _build_saving(saving, general, mixed_precision, main)
    errors += saving_errors

    # ---- sampling cadence + sample.toml ------------------------------------
    sample_cfg = _build_sample(sample, main)

    # ---- [monitoring] ------------------------------------------------------
    monitoring_cfg = _build_monitoring(logging_group, saving)

    # ---- extra_args (free-form top-level injection) ------------------------
    for key, value in extra.items():
        if _blank(key):
            continue
        if isinstance(value, str):
            main[key] = _toml_literal(value)
        else:
            main[key] = value

    # ---- dataset.toml ------------------------------------------------------
    dataset_cfg, subsets_out, dataset_errors = _build_dataset(ds_general, bucket, subsets_in)
    errors += dataset_errors

    # ---- warmup_ratio -> warmup_steps (needs dataset + epochs) -------------
    warmup_ratio = _as_float(optimizer.get("warmup_ratio"))
    if warmup_ratio is not None and "warmup_steps" not in main:
        steps = _calculate_steps(subsets_out, main["epochs"], grad_acc, num_gpus)
        main["warmup_steps"] = round(steps * warmup_ratio)

    # attach tables (process.py fills in dataset/sample paths)
    main["model"] = model_cfg
    if adapter_cfg is not None:
        main["adapter"] = adapter_cfg
    main["optimizer"] = optimizer_cfg
    main["monitoring"] = monitoring_cfg

    # Multi-caption: build a captions.json per subset from .txt (tags) + .caption
    # (NL). With online_captions on and enable_random_caption OFF, diffusion-pipe
    # trains one example per caption variant, so images with both files are seen
    # twice per epoch. Only when the whole payload is valid, to avoid writing
    # files on a failed validation.
    if _as_bool(caption_args.get("combine_txt_caption")) and not errors:
        from utils.captions import build_captions_json

        built = sum(1 for subset in subsets_out if build_captions_json(Path(subset["path"])))
        if built:
            dataset_cfg["online_captions"] = True
            # enable_random_caption stays off -> train on every variant.

    tags = _collect_tags(subsets_out)

    passed = len(errors) == 0
    return passed, errors, main, dataset_cfg, sample_cfg, tags


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _validate_path(value, label, errors, must_be_file=False):
    path = Path(value)
    if not path.exists():
        errors.append(f"{label} '{value}' does not exist")
        return None
    if must_be_file and not path.is_file():
        errors.append(f"{label} '{value}' is not a file")
        return None
    return path.as_posix()


def _build_model(group: dict, model_dtype: str):
    errors: list[str] = []
    cfg: dict = {"type": "anima"}

    transformer = _first(group, "transformer_path", "pretrained_model_name_or_path")
    llm = _first(group, "llm_path", "qwen3")
    vae = _first(group, "vae_path", "vae")

    if _blank(transformer):
        errors.append("transformer_path (base model) is required")
    else:
        resolved = _validate_path(transformer, "transformer_path", errors, must_be_file=True)
        if resolved:
            cfg["transformer_path"] = resolved
    if _blank(llm):
        errors.append("llm_path (Qwen3 text encoder) is required")
    else:
        # llm_path may be a file (Qwen3 safetensors) or a directory (generic LLM).
        resolved = _validate_path(llm, "llm_path", errors)
        if resolved:
            cfg["llm_path"] = resolved
    if _blank(vae):
        errors.append("vae_path is required")
    else:
        resolved = _validate_path(vae, "vae_path", errors, must_be_file=True)
        if resolved:
            cfg["vae_path"] = resolved

    cfg["dtype"] = model_dtype
    transformer_dtype = _first(group, "transformer_dtype")
    if transformer_dtype:
        cfg["transformer_dtype"] = DTYPE_MAP.get(str(transformer_dtype).lower(), model_dtype)

    method = _first(group, "timestep_sample_method", "timestep_sampling")
    if method:
        method = str(method).lower()
        if method not in VALID_TIMESTEP_METHODS:
            errors.append(
                f"timestep_sample_method '{method}' is not supported by Anima "
                f"(use one of {sorted(VALID_TIMESTEP_METHODS)})"
            )
        else:
            cfg["timestep_sample_method"] = method

    sigmoid_scale = _as_float(group.get("sigmoid_scale"))
    if sigmoid_scale is not None:
        cfg["sigmoid_scale"] = sigmoid_scale

    # llm_adapter_lr defaults to 0 (freeze the Qwen3->DiT adapter) unless the
    # user explicitly opts in.
    if not _blank(group.get("llm_adapter_lr")):
        cfg["llm_adapter_lr"] = _as_float(group.get("llm_adapter_lr"), 0.0)
    else:
        cfg["llm_adapter_lr"] = 0

    shift = _as_float(group.get("shift"))
    if shift is not None:
        cfg["shift"] = shift
    elif _as_bool(group.get("flux_shift")):
        cfg["flux_shift"] = True

    multiscale = _as_float(group.get("multiscale_loss_weight"))
    if multiscale is not None:
        cfg["multiscale_loss_weight"] = multiscale
    contrastive = _as_float(group.get("contrastive_flow_lambda"))
    if contrastive:
        cfg["contrastive_flow_lambda"] = contrastive
    if group.get("cache_text_embeddings") is not None:
        cfg["cache_text_embeddings"] = _as_bool(group.get("cache_text_embeddings"), True)

    for lr_key in ("self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr"):
        if not _blank(group.get(lr_key)):
            cfg[lr_key] = _as_float(group.get(lr_key))

    return cfg, errors


def _build_adapter(group: dict, model_dtype: str):
    """Return (adapter_cfg or None, errors). ``None`` means full fine-tune."""
    errors: list[str] = []
    adapter_type = _first(group, "type", "algo")
    if _blank(adapter_type):
        return None, errors
    adapter_type = str(adapter_type).lower()
    if adapter_type in ("none", "full", "fft", "full_finetune", "full fine-tune"):
        return None, errors
    if adapter_type not in ADAPTER_TYPES:
        errors.append(
            f"adapter type '{adapter_type}' is not supported by diffusion-pipe "
            f"(use 'lora', 'lokr', or 'none' for full fine-tune)"
        )
        return None, errors

    cfg: dict = {"type": adapter_type}
    if not _blank(group.get("alpha")) or not _blank(group.get("network_alpha")):
        errors.append(
            "network_alpha is not supported: diffusion-pipe forces alpha = rank. "
            "Remove it from the config."
        )
    cfg["rank"] = _as_int(_first(group, "rank", "network_dim"), 32) or 32
    dropout = _as_float(_first(group, "dropout", "network_dropout"))
    if dropout:
        cfg["dropout"] = dropout
    cfg["dtype"] = DTYPE_MAP.get(str(_first(group, "dtype", default=model_dtype)).lower(), model_dtype)

    if adapter_type == "lokr":
        factor = _as_int(group.get("factor"))
        cfg["factor"] = factor if factor is not None else -1
        for flag in ("use_tucker", "decompose_both", "rank_dropout_scale", "include_conv"):
            if _as_bool(group.get(flag)):
                cfg[flag] = True
        for num_key in ("rank_dropout", "module_dropout"):
            val = _as_float(group.get(num_key))
            if val:
                cfg[num_key] = val

    init_from = _first(group, "init_from_existing", "network_weights")
    if init_from:
        # Path may be an adapter run dir; existence is best-effort here.
        cfg["init_from_existing"] = Path(init_from).as_posix()

    return cfg, errors


def _build_optimizer(group: dict, main: dict):
    errors: list[str] = []
    cfg: dict = {}

    opt_type = _first(group, "optimizer_type", "type", default="adamw_optimi")
    cfg["type"] = OPTIMIZER_NAME_MAP.get(str(opt_type).lower(), opt_type)

    lr = _as_float(_first(group, "lr", "learning_rate", "unet_lr"))
    if lr is None:
        errors.append("learning_rate is required")
    else:
        cfg["lr"] = lr

    # Inlined optimizer sub-args (betas, weight_decay, eps, ...).
    sub = group.get("optimizer_args", {}) or {}
    if isinstance(sub, dict):
        for key, value in sub.items():
            if _blank(value):
                continue
            cfg[key] = value

    # top-level training keys derived from the optimizer group
    max_grad_norm = _as_float(_first(group, "max_grad_norm", "gradient_clipping"))
    if max_grad_norm is not None:
        main["gradient_clipping"] = max_grad_norm

    warmup_steps = _as_int(_first(group, "warmup_steps", "lr_warmup_steps"))
    if warmup_steps is not None:
        main["warmup_steps"] = warmup_steps

    scheduler = _first(group, "lr_scheduler")
    if scheduler:
        scheduler = str(scheduler).lower()
        if scheduler in VALID_LR_SCHEDULERS and scheduler != "constant":
            main["lr_scheduler"] = scheduler
        elif scheduler not in VALID_LR_SCHEDULERS:
            errors.append(
                f"lr_scheduler '{scheduler}' is not supported "
                f"(use one of {sorted(VALID_LR_SCHEDULERS)})"
            )

    force_constant = _as_float(group.get("force_constant_lr"))
    if force_constant is not None:
        main["force_constant_lr"] = force_constant

    # Huber / smooth-L1 loss (Anima reads these top-level keys).
    loss_type = str(_first(group, "loss_type", default="")).lower()
    if loss_type in ("huber", "smooth_l1"):
        c = _as_float(_first(group, "huber_c", "huber_delta", "smooth_l1_beta"))
        if c is not None:
            main["huber_delta" if loss_type == "huber" else "smooth_l1_beta"] = c

    return cfg, errors


def _build_saving(saving: dict, general: dict, mixed_precision, main: dict):
    errors: list[str] = []

    output_dir = _first(saving, "output_dir")
    if _blank(output_dir):
        errors.append("output_dir is required")
    else:
        path = Path(output_dir)
        if not path.exists():
            if not path.parent.exists():
                errors.append(f"Parent path for output_dir '{path.parent}' does not exist")
            else:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    main["output_dir"] = path.as_posix()
                except OSError as exc:
                    errors.append(f"Could not create output_dir: {exc}")
        else:
            main["output_dir"] = path.as_posix()

    main["output_name"] = _first(saving, "output_name", default="model")

    save_epochs = _as_int(saving.get("save_every_n_epochs"))
    save_steps = _as_int(saving.get("save_every_n_steps"))
    save_examples = _as_int(saving.get("save_every_n_examples"))
    if save_epochs:
        main["save_every_n_epochs"] = save_epochs
    if save_steps:
        main["save_every_n_steps"] = save_steps
    if save_examples:
        main["save_every_n_examples"] = save_examples
    if not (save_epochs or save_steps or save_examples):
        # diffusion-pipe asserts at least one; default to every epoch.
        main["save_every_n_epochs"] = 1

    save_precision = _first(saving, "save_precision", default=mixed_precision)
    main["save_dtype"] = DTYPE_MAP.get(str(save_precision).lower(), "bfloat16")

    ckpt_epochs = _as_int(saving.get("checkpoint_every_n_epochs"))
    ckpt_minutes = _as_int(saving.get("checkpoint_every_n_minutes"))
    if ckpt_epochs:
        main["checkpoint_every_n_epochs"] = ckpt_epochs
    elif ckpt_minutes:
        main["checkpoint_every_n_minutes"] = ckpt_minutes

    return errors


def _build_sample(sample: dict, main: dict):
    sample_epochs = _as_int(sample.get("sample_every_n_epochs"))
    sample_steps = _as_int(sample.get("sample_every_n_steps"))
    if sample_epochs:
        main["sample_every_n_epochs"] = sample_epochs
    if sample_steps:
        main["sample_every_n_steps"] = sample_steps
    if _as_bool(sample.get("sample_at_first")):
        main["sample_at_first"] = True

    prompts_in = sample.get("prompts", []) or []
    prompts: list[dict] = []
    for entry in prompts_in:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                prompts.append({"prompt": text})
            continue
        if not isinstance(entry, dict):
            continue
        text = entry.get("prompt", "")
        if _blank(text):
            continue
        prompt = {"prompt": text}
        neg = entry.get("negative_prompt")
        if not _blank(neg):
            prompt["negative_prompt"] = neg
        prompts.append(prompt)

    if not prompts:
        # No prompts -> no sampling; drop cadence keys to avoid a dangling ref.
        main.pop("sample_every_n_epochs", None)
        main.pop("sample_every_n_steps", None)
        main.pop("sample_at_first", None)
        return None

    cfg = {
        "width": _as_int(sample.get("width"), 1024) or 1024,
        "height": _as_int(sample.get("height"), 1024) or 1024,
        "num_inference_steps": _as_int(_first(sample, "num_inference_steps", "steps"), 32) or 32,
        "guidance_scale": _as_float(_first(sample, "guidance_scale", "cfg"), 4.0),
        "seed": _as_int(sample.get("seed"), 42),
        "prompts": prompts,
    }
    return cfg


def _build_monitoring(logging_group: dict, saving: dict):
    cfg = {"enable_wandb": _as_bool(logging_group.get("enable_wandb"))}
    if cfg["enable_wandb"]:
        cfg["wandb_api_key"] = _first(logging_group, "wandb_api_key", default="")
        cfg["wandb_tracker_name"] = _first(logging_group, "wandb_tracker_name", default="")
        run_name = _first(logging_group, "wandb_run_name")
        if _blank(run_name):
            run_name = _first(saving, "output_name")
        if run_name:
            cfg["wandb_run_name"] = run_name
    return cfg


def _build_dataset(general: dict, bucket: dict, subsets_in: list):
    errors: list[str] = []
    cfg: dict = {}

    resolutions = general.get("resolutions", general.get("resolution"))
    cfg["resolutions"] = _normalize_resolutions(resolutions)

    cfg["enable_ar_bucket"] = _as_bool(bucket.get("enable_ar_bucket", bucket.get("enable_bucket", True)), True)
    cfg["min_ar"] = _as_float(bucket.get("min_ar"), 0.5)
    cfg["max_ar"] = _as_float(bucket.get("max_ar"), 2.0)
    cfg["num_ar_buckets"] = _as_int(bucket.get("num_ar_buckets"), 7) or 7
    cfg["frame_buckets"] = [1]

    # Global caption knobs.
    shuffle_num = _as_int(general.get("cache_shuffle_num"))
    if shuffle_num is None and _as_bool(general.get("shuffle_caption")):
        shuffle_num = 10  # sensible default number of pre-shuffled variants
    if shuffle_num:
        cfg["cache_shuffle_num"] = shuffle_num
        delimiter = _first(general, "cache_shuffle_delimiter")
        if delimiter:
            cfg["cache_shuffle_delimiter"] = delimiter
    keep_tokens = _as_int(general.get("keep_tokens"))
    if keep_tokens:
        cfg["keep_tokens"] = keep_tokens
    keep_sep = _first(general, "keep_tokens_separator")
    if keep_sep:
        cfg["keep_tokens_separator"] = keep_sep
    if _as_bool(general.get("enable_random_caption")):
        cfg["enable_random_caption"] = True
    if general.get("skip_empty_caption") is not None:
        cfg["skip_empty_caption"] = _as_bool(general.get("skip_empty_caption"), True)
    if _as_bool(general.get("online_captions")):
        cfg["online_captions"] = True

    directories: list[dict] = []
    if not subsets_in:
        errors.append("At least one dataset directory (subset) is required")
    for index, subset in enumerate(subsets_in):
        path_value = _first(subset, "path", "image_dir")
        if _blank(path_value):
            errors.append(f"[subset {index}] image directory path is required")
            continue
        path = Path(path_value)
        if not path.is_dir():
            errors.append(f"[subset {index}] image directory '{path_value}' does not exist")
            continue
        entry = {"path": path.as_posix(), "num_repeats": _as_int(subset.get("num_repeats"), 1) or 1}
        mask = _first(subset, "mask_path")
        if mask:
            mask_path = Path(mask)
            if not mask_path.is_dir():
                errors.append(f"[subset {index}] mask_path '{mask}' does not exist")
            else:
                entry["mask_path"] = mask_path.as_posix()
        directories.append(entry)

    cfg["directory"] = directories
    return cfg, directories, errors


def _normalize_resolutions(value):
    if value is None:
        return [1024]
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append([int(item[0]), int(item[1])])
            elif not _blank(item):
                out.append(int(item))
        return out or [1024]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return [1024]


# --------------------------------------------------------------------------- #
# Step counting (for warmup_ratio) + tag counting
# --------------------------------------------------------------------------- #
def _count_images(subsets: list) -> int:
    total = 0
    for subset in subsets:
        directory = Path(subset["path"])
        if not directory.is_dir():
            continue
        count = sum(
            1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        total += count * subset.get("num_repeats", 1)
    return total


def _calculate_steps(subsets: list, epochs: int, grad_acc: int, num_gpus: int) -> int:
    images = _count_images(subsets)
    if images == 0:
        return 0
    per_epoch = math.ceil(images / max(grad_acc, 1) / max(num_gpus, 1))
    return per_epoch * max(epochs, 1)


def _collect_tags(subsets: list) -> dict:
    tags: dict[str, int] = {}
    seen: set[str] = set()
    for subset in subsets:
        directory = Path(subset["path"])
        if directory.as_posix() in seen or not directory.is_dir():
            continue
        seen.add(directory.as_posix())
        for caption_file in directory.glob("*.txt"):
            _read_tags(caption_file, tags)
    return dict(sorted(tags.items(), key=lambda kv: kv[1], reverse=True))


def _read_tags(path: Path, tags: dict) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for tag in content.replace(", ", ",").split(","):
        tag = tag.strip()
        if not tag:
            continue
        tags[tag] = tags.get(tag, 0) + 1
