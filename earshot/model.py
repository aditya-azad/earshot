from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    processor: ProcessorMixin


def load_model(
    model_id: str,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    processor = AutoProcessor.from_pretrained(model_id)
    if processor is None:
        raise ValueError(f"Could not load processor for model '{model_id}'")

    errors: list[str] = []
    model: PreTrainedModel | None = None
    for loader in (AutoModelForSpeechSeq2Seq, AutoModelForCTC):
        try:
            model = loader.from_pretrained(model_id, torch_dtype=dtype)
            break
        except (ValueError, EnvironmentError) as exc:
            errors.append(f"{loader.__name__}: {exc}")

    if model is None:
        raise ValueError(
            f"'{model_id}' is not a supported speech-to-text model. "
            f"Attempted loaders failed:\n" + "\n".join(errors)
        )

    if device is not None:
        model = model.to(device)
    model.eval()
    return LoadedModel(model=model, processor=processor)
