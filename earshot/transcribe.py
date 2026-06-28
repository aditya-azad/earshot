from __future__ import annotations

import numpy as np
import torch

from .config import SAMPLE_RATE
from .model import LoadedModel


def transcribe(loaded: LoadedModel, audio: np.ndarray) -> str:
    model = loaded.model
    processor = loaded.processor
    params = next(model.parameters())
    device = params.device
    dtype = params.dtype

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")  # type: ignore[unexpected-keyword]
    inputs = {
        k: v.to(device=device, dtype=dtype)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        if hasattr(model, "generate"):
            predicted_ids = model.generate(**inputs)  # type: ignore[not-callable]
        else:
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()
