#!/usr/bin/env python3
"""One-shot rebuild of every feaweld documentation asset.

Runs the four generator scripts in sequence:
    1. scripts/generate_docs_images.py         (14 legacy round-0 SVGs)
    2. scripts/generate_docs_mermaid.py        (8 mermaid .mmd files)
    3. scripts/generate_docs_concept_images.py (23 new concept SVG+PNG pairs)
    4. scripts/generate_docs_animations.py     (9 animation GIF+MP4 pairs)

Usage:
    python scripts/build_docs_assets.py
    python scripts/build_docs_assets.py --skip-animations
    python scripts/build_docs_assets.py --only mermaid concept_images
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT / "src"))


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / file_name)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run_group(label: str, module_name: str, file_name: str) -> None:
    print(f"\n=== {label} ===")
    start = time.monotonic()
    try:
        mod = _load(module_name, file_name)
        mod.main()
    except Exception as exc:
        print(f"  !! {label} failed: {type(exc).__name__}: {exc}")
        raise
    print(f"  done in {time.monotonic() - start:.1f}s")


_GROUPS = {
    "legacy_images": ("Legacy round-0 images",
                      "docs_legacy_images",
                      "generate_docs_images.py"),
    "mermaid": ("Mermaid diagrams",
                "docs_mermaid",
                "generate_docs_mermaid.py"),
    "concept_images": ("Advanced concept images (23)",
                       "docs_concept_images",
                       "generate_docs_concept_images.py"),
    "animations": ("Animations (9)",
                   "docs_animations",
                   "generate_docs_animations.py"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*",
                        choices=list(_GROUPS.keys()),
                        help="Subset of groups to run")
    parser.add_argument("--skip-animations", action="store_true",
                        help="Skip the (slower) animation group")
    args = parser.parse_args(argv)

    groups = list(_GROUPS.keys())
    if args.only:
        groups = args.only
    if args.skip_animations and "animations" in groups:
        groups.remove("animations")

    total = time.monotonic()
    for key in groups:
        label, modname, filename = _GROUPS[key]
        if not (_SCRIPTS / filename).exists():
            print(f"[skip] {label}: {filename} not found")
            continue
        _run_group(label, modname, filename)
    print(f"\nAll groups done in {time.monotonic() - total:.1f}s.")


if __name__ == "__main__":
    main()
