#!/usr/bin/env python3
"""Terminal push-to-talk speech-to-text tester.

Hold the trigger key (default: space) to record from the microphone;
release it to run the loaded model and print the transcribed text.

Usage:
    earshot-test openai/whisper-tiny
    earshot-test openai/whisper-tiny --key space
    earshot-test facebook/wav2vec2-base-960h

Press Esc to quit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
import torch
from pynput import keyboard

from .model import LoadedModel, load_model

SAMPLE_RATE = 16000


def parse_key(name: str) -> keyboard.Key | keyboard.KeyCode:
    name = name.lower()
    special = {
        "space": keyboard.Key.space,
        "enter": keyboard.Key.enter,
        "esc": keyboard.Key.esc,
        "tab": keyboard.Key.tab,
        "shift": keyboard.Key.shift,
    }
    if name in special:
        return special[name]
    return keyboard.KeyCode.from_char(name)


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
        print("[recording] ", end="", flush=True)

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


def main() -> None:
    default_model = os.environ.get("EARSHOT_MODEL", "openai/whisper-tiny")

    parser = argparse.ArgumentParser(
        description="Hold a key to record speech and print the model's transcription."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        help=f"HuggingFace model id (default: {default_model}, "
        "or $EARSHOT_MODEL if set)",
    )
    parser.add_argument(
        "--key",
        default="space",
        help="Key to hold while recording (default: space). "
        "Use a single char like 'r' or a name like 'space', 'enter'.",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    print(f"Loading '{args.model}' on {device} ({dtype})...")
    loaded = load_model(args.model, device=device, dtype=dtype)
    print("Model loaded.")

    target_key = parse_key(args.key)
    recorder = Recorder()
    recorder.start_stream()
    state = {"recording": False, "done": False}

    key_label = args.key
    print(
        f"\nHold {key_label!r} to record, release to transcribe. "
        f"Press Esc to quit.\n"
    )

    def on_press(key) -> None:
        if state["done"]:
            return
        if key == target_key and not state["recording"]:
            state["recording"] = True
            recorder.start()
            print("[recording] ", end="", flush=True)

    def on_release(key):
        if key == keyboard.Key.esc:
            state["done"] = True
            if state["recording"]:
                recorder.stop()
                state["recording"] = False
            return False
        if key == target_key and state["recording"]:
            state["recording"] = False
            start = time.perf_counter()
            audio = recorder.stop()
            if audio is None or audio.size == 0:
                print("\n(no audio captured)")
                return
            duration = len(audio) / SAMPLE_RATE
            print(f"\n(transcribing {duration:.1f}s)...", flush=True)
            text = transcribe(loaded, audio)
            elapsed = time.perf_counter() - start
            print(f"=> {text}   [{elapsed:.2f}s]\n")

    with keyboard.Listener(
        on_press=on_press, on_release=on_release, suppress=True
    ) as listener:
        listener.join()

    recorder.stop_stream()
    print("Bye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
