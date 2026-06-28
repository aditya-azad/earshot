#!/usr/bin/env python3
"""Terminal push-to-talk speech-to-text tester.

Hold the trigger key (default: space) to record from the microphone;
release it to run the loaded model and print the transcribed text.

Usage:
    test-stt openai/whisper-tiny
    test-stt openai/whisper-tiny --key space
    test-stt facebook/wav2vec2-base-960h

Press Esc to quit.
"""

from __future__ import annotations

import time

from pynput import keyboard

from .audio import Recorder
from .cli import build_parser, load_for_cli, run_entry_point
from .config import SAMPLE_RATE
from .keys import normalize_key, parse_combo
from .transcribe import transcribe


def main() -> None:
    key_help = (
        "Key to hold while recording (default: space). "
        "Use a single char like 'r' or a name like 'space', 'enter'."
    )
    parser = build_parser(
        "Hold a key to record speech and print the model's transcription.",
        key_default="space",
        key_help=key_help,
    )
    args = parser.parse_args()

    loaded = load_for_cli(args)

    combo = parse_combo(args.key)
    recorder = Recorder()
    recorder.start_stream()
    done = False

    key_label = args.key
    print(
        f"\nHold {key_label!r} to record, release to transcribe. "
        "Press Esc to quit.\n"
    )

    def on_press(key) -> None:
        nonlocal done
        if done:
            return
        if normalize_key(key, combo) in combo and not recorder.recording:
            recorder.start()
            print("[recording] ", end="", flush=True)

    def on_release(key):
        nonlocal done
        if key == keyboard.Key.esc:
            done = True
            if recorder.recording:
                recorder.stop()
            return False  # noqa: FBT001  – pynput uses False to stop the listener
        if normalize_key(key, combo) in combo and recorder.recording:
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
        on_press=on_press, on_release=on_release, suppress=True  # type: ignore[bad-argument-type]
    ) as listener:
        listener.join()

    recorder.stop_stream()
    print("Bye.")


if __name__ == "__main__":
    run_entry_point(main)
