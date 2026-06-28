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

import torch
from pynput import keyboard

from .audio import SAMPLE_RATE, Recorder, transcribe
from .model import load_model


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


def main() -> None:
    default_model = os.environ.get("EARSHOT_MODEL", "openai/whisper-large-v3")

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
    done = False

    key_label = args.key
    print(
        f"\nHold {key_label!r} to record, release to transcribe. Press Esc to quit.\n"
    )

    def on_press(key) -> None:
        nonlocal done
        if done:
            return
        if key == target_key and not recorder.recording:
            recorder.start()
            print("[recording] ", end="", flush=True)

    def on_release(key):
        nonlocal done
        if key == keyboard.Key.esc:
            done = True
            if recorder.recording:
                recorder.stop()
            return False
        if key == target_key and recorder.recording:
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
