"""Build a diffusion-pipe ``captions.json`` from ``.txt`` (tags) + ``.caption`` (NL).

diffusion-pipe's dataset reads captions from ``<image>.txt`` only, unless a
``captions.json`` sits in the directory and ``online_captions = true``. This
lets a dataset with both booru tags (``.txt``) and a natural-language caption
(``.caption``) train on both.

We give each image a list of caption variants: ``[tags]``, ``[nl]``, or
``[tags, nl]``. With ``enable_random_caption`` OFF (the default), diffusion-pipe
creates one training example **per variant** each epoch, so:

- image with only ``.txt``  -> seen once  per epoch (tags)
- image with ``.txt`` + ``.caption`` -> seen twice per epoch (once tags, once NL)

The JSON key must match how diffusion-pipe enumerates files
(``str(self.path.glob('*'))``); we iterate the same directory, so ``str(image)``
matches exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
CAPTIONS_JSON = "captions.json"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def build_captions_json(directory: Path) -> int:
    """Write ``captions.json`` into *directory*. Returns the number of images written.

    Each image maps to its available captions: ``[tags]`` / ``[nl]`` / ``[tags, nl]``.
    """
    if not directory.is_dir():
        return 0

    images = sorted(f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    data: dict[str, list[str]] = {}

    for image in images:
        tags = _read(image.with_suffix(".txt"))
        nl = _read(image.with_suffix(".caption"))
        variants = [c for c in (tags, nl) if c]
        # de-dupe (e.g. identical tags/nl) while preserving order
        seen: set[str] = set()
        variants = [c for c in variants if not (c in seen or seen.add(c))]
        if variants:
            data[str(image)] = variants

    if not data:
        return 0

    (directory / CAPTIONS_JSON).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(data)
