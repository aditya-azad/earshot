from __future__ import annotations

import torch

DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_KEY = "shift_r"
SAMPLE_RATE = 16000

COLOR_IDLE = (90, 90, 90, 255)
COLOR_RECORDING = (220, 70, 70, 255)
COLOR_TRANSCRIBING = (70, 130, 220, 255)


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype
