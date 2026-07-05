"""Serialize the validated config dicts into diffusion-pipe's three TOMLs.

``write_configs`` writes ``runtime_store/{main,dataset,sample}.toml`` and wires
the main config to the other two by **absolute** path (diffusion-pipe resolves
``dataset``/``sample`` relative to its own cwd, so absolute is required).
"""
from __future__ import annotations

from pathlib import Path

from utils.toml_writer import dumps

RUNTIME_STORE = Path("runtime_store")


def _ensure_store() -> Path:
    RUNTIME_STORE.mkdir(exist_ok=True)
    return RUNTIME_STORE


def write_configs(main: dict, dataset: dict, sample: dict | None):
    """Write the three TOMLs. Returns (main_path, dataset_path, sample_path).

    ``sample_path`` is ``None`` when there are no sample prompts. The passed-in
    ``main`` dict is updated in place with the resolved ``dataset``/``sample``
    paths before serialization.
    """
    store = _ensure_store()
    main_path = (store / "main.toml").resolve()
    dataset_path = (store / "dataset.toml").resolve()
    sample_path = (store / "sample.toml").resolve() if sample else None

    # Cross-reference by absolute path.
    main = dict(main)
    main["dataset"] = dataset_path.as_posix()
    if sample_path is not None:
        main["sample"] = sample_path.as_posix()
    else:
        main.pop("sample", None)

    dataset_path.write_text(dumps(dataset), encoding="utf-8")
    if sample_path is not None:
        sample_path.write_text(dumps(sample), encoding="utf-8")
    main_path.write_text(dumps(main), encoding="utf-8")

    return main_path, dataset_path, sample_path
