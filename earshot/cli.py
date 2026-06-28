from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from .config import DEFAULT_MODEL, resolve_device_dtype
from .model import LoadedModel, load_model


def build_parser(
    description: str, *, key_default: str, key_help: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    default_model = os.environ.get("EARSHOT_MODEL", DEFAULT_MODEL)
    parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        help=f"HuggingFace model id (default: {default_model}, "
        "or $EARSHOT_MODEL if set)",
    )
    parser.add_argument(
        "--key",
        default=key_default,
        help=key_help,
    )
    return parser


def load_for_cli(args: argparse.Namespace) -> LoadedModel:
    device, dtype = resolve_device_dtype()
    print(f"Loading '{args.model}' on {device} ({dtype})...")
    loaded = load_model(args.model, device=device, dtype=dtype)
    print("Model loaded.")
    return loaded


def run_entry_point(main: Callable[[], None]) -> None:
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
