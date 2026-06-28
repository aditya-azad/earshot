from __future__ import annotations

import numpy as np
import sounddevice as sd
import torch

from .model import LoadedModel

SAMPLE_RATE = 16000


def transcribe(loaded: LoadedModel, audio: np.ndarray) -> str:
    model = loaded.model
    processor = loaded.processor
    params = next(model.parameters())
    device = params.device
    dtype = params.dtype

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=dtype)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        if hasattr(model, "generate"):
            predicted_ids = model.generate(**inputs)
        else:
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


class Recorder:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self.recording = False
        self._stream: sd.InputStream | None = None

    def start_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def start(self) -> None:
        if self.recording:
            return
        self._chunks = []
        self.recording = True

    def _callback(self, indata: np.ndarray, frames, time, status) -> None:  # noqa: A002
        if self.recording:
            self._chunks.append(indata.copy())

    def stop(self) -> np.ndarray | None:
        if not self.recording:
            return None
        self.recording = False
        if not self._chunks:
            return None
        return np.concatenate(self._chunks, axis=0).flatten()
