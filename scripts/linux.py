#!/usr/bin/env python3
"""Linux system-tray speech-to-text application.

Runs in the background with an icon in the topbar / notification area.
The microphone stream is started on launch.  Hold the trigger key
combination (default: ``shift_r`` — the right Shift key) to record;
release it to transcribe and type the result at the cursor.

The keyboard is **never** grabbed: all key events keep reaching other
applications, so you can type normally.  Recording only starts when the
full combination is held, which prevents accidental triggers while
typing.  (The combination keys themselves still pass through to other
apps — pick modifiers / named keys that won't interfere with your
workflow.)

Right-side modifiers (``shift_r``, ``ctrl_r``, ``alt_r``) are matched
specifically, so ``shift_r`` triggers only on the right Shift key, not
the left.  The generic names (``shift``, ``ctrl``, ``alt``) match either
side.

Requires the GNOME ``AppIndicator`` extension (or an equivalent
KStatusNotifierItem support) for the tray icon, ``notify-send`` for
desktop notifications, and ``xdotool`` / ``wtype`` for typing.

Usage:
    uv run scripts/linux.py openai/whisper-tiny
    uv run scripts/linux.py openai/whisper-tiny --key shift_r
    uv run scripts/linux.py openai/whisper-tiny --key ctrl+space
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import sounddevice as sd
import torch
from pynput import keyboard
from pystray import Icon, Menu, MenuItem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import LoadedModel, load_model
from src.notify import notify
from src.typing import TextTyper, make_typer

SAMPLE_RATE = 16000

COLOR_IDLE = (90, 90, 90, 255)
COLOR_RECORDING = (220, 70, 70, 255)
COLOR_TRANSCRIBING = (70, 130, 220, 255)


# ── helpers ──────────────────────────────────────────────────────────


def make_icon(color: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([24, 6, 40, 34], radius=8, fill=color)
    draw.arc([18, 18, 46, 48], 180, 360, fill=color, width=3)
    draw.rectangle([30, 38, 34, 50], fill=color)
    draw.rectangle([22, 50, 42, 54], fill=color)
    return img


# Named modifier keys, including right-side variants.  The right-side
# ones are kept distinct from their left-side counterparts so a combo can
# require a specific side (e.g. only the right Shift key).
_MODIFIERS = {
    name: getattr(keyboard.Key, name)
    for name in (
        "shift",
        "shift_r",
        "ctrl",
        "ctrl_r",
        "alt",
        "alt_r",
        "cmd",
        "cmd_r",
        "alt_gr",
    )
}

# Left/right modifier pairs.  When a combo pins the right member, the two
# sides are matched distinctly; otherwise both collapse to the left form.
_MODIFIER_PAIRS = [
    (keyboard.Key.shift, keyboard.Key.shift_r),
    (keyboard.Key.ctrl, keyboard.Key.ctrl_r),
    (keyboard.Key.alt, keyboard.Key.alt_r),
    (keyboard.Key.cmd, keyboard.Key.cmd_r),
]

# Friendly names for the tray menu.
_DISPLAY_NAMES = {
    "shift_r": "Right Shift",
    "shift": "Shift",
    "ctrl_r": "Right Ctrl",
    "ctrl": "Ctrl",
    "alt_r": "Right Alt",
    "alt": "Alt",
    "cmd_r": "Right Super",
    "cmd": "Super",
    "alt_gr": "AltGr",
    "space": "Space",
    "enter": "Enter",
    "tab": "Tab",
    "esc": "Esc",
}


def pretty_label(spec: str) -> str:
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    return " + ".join(_DISPLAY_NAMES.get(p, p) for p in parts)


def parse_combo(spec: str) -> set:
    """Parse a key combination such as ``shift_r`` or ``ctrl+space``.

    Single characters are treated as literal keys (e.g. ``r``); anything
    else is treated as a named key (e.g. ``space``, ``shift_r``, ``enter``).
    Right-side modifiers (``shift_r``, ``ctrl_r``, ``alt_r``) are kept
    distinct from the left ones, so ``shift_r`` matches only the right
    Shift key.  Other named keys are normalised to their virtual-key code
    so they still match while modifiers are held.
    """
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty key combination: {spec!r}")
    keys: set = set()
    for p in parts:
        if p in _MODIFIERS:
            keys.add(_MODIFIERS[p])
        elif len(p) == 1:
            keys.add(keyboard.KeyCode.from_char(p))
        else:
            try:
                k = keyboard.Key[p]
            except KeyError as e:
                raise ValueError(f"unknown key: {p!r}") from e
            if k.value.vk is not None:
                keys.add(keyboard.KeyCode.from_vk(k.value.vk))
            else:
                keys.add(k)
    return keys


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


# ── audio recorder ───────────────────────────────────────────────────


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


# ── tray application ─────────────────────────────────────────────────


class TrayApp:
    def __init__(
        self,
        loaded: LoadedModel,
        combo: set,
        key_label: str,
        typer: TextTyper,
    ) -> None:
        self.loaded = loaded
        self.combo = combo
        self.key_label = key_label
        self.typer = typer
        self.pressed: set = set()
        self.recorder = Recorder()
        self.recording = False
        self.transcribing = False
        self.listening = True
        self.last_text = ""
        self.listener: keyboard.Listener | None = None
        self.icon: Icon | None = None

    # -- icon / menu helpers -------------------------------------------

    def _set_icon_color(self, color: tuple[int, int, int, int]) -> None:
        if self.icon is not None:
            self.icon.icon = make_icon(color)

    def _status_text(self, _icon: Icon) -> str:
        if self.transcribing:
            return "Transcribing…"
        if self.recording:
            return "Recording…"
        if self.listening:
            return "Listening"
        return "Paused"

    def _last_text_text(self, _icon: Icon) -> str:
        if self.last_text:
            truncated = self.last_text[:50]
            if len(self.last_text) > 50:
                truncated += "…"
            return f"Last: {truncated}"
        return "No transcription yet"

    def _toggle_label(self, _icon: Icon) -> str:
        return "Pause listening" if self.listening else "Resume listening"

    # -- keyboard callbacks --------------------------------------------

    def _normalize(self, key) -> object:
        # Like pynput's Listener.canonical, but keeps left/right modifiers
        # distinct when the combo pins a specific side (e.g. shift_r), so
        # only that side matches.  Generic modifier names collapse to the
        # left form and match either side.
        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return keyboard.KeyCode.from_char(key.char.lower())
            return key
        if isinstance(key, keyboard.Key):
            for left, right in _MODIFIER_PAIRS:
                if key in (left, right):
                    return key if right in self.combo else left
            if key in _MODIFIERS.values():
                return key
            if key.value.vk is not None:
                return keyboard.KeyCode.from_vk(key.value.vk)
        return key

    def _on_press(self, key) -> None:
        if not self.listening or self.transcribing:
            return
        self.pressed.add(self._normalize(key))
        if not self.recording and self.combo.issubset(self.pressed):
            self.recording = True
            self.recorder.start()
            self._set_icon_color(COLOR_RECORDING)

    def _on_release(self, key) -> None:
        self.pressed.discard(self._normalize(key))
        if self.recording and not self.combo.issubset(self.pressed):
            self.recording = False
            self._set_icon_color(COLOR_TRANSCRIBING)
            audio = self.recorder.stop()
            if audio is None or audio.size == 0:
                self._set_icon_color(COLOR_IDLE)
                return
            self.transcribing = True
            if self.icon is not None:
                self.icon.update_menu()
            threading.Thread(
                target=self._transcribe_async,
                args=(audio,),
                daemon=True,
            ).start()

    def _transcribe_async(self, audio: np.ndarray) -> None:
        try:
            text = transcribe(self.loaded, audio)
            if text:
                self.last_text = text
                try:
                    self.typer.type_text(text)
                except Exception as exc:
                    notify("Earshot — typing error", str(exc))
                else:
                    notify("Earshot — transcribed", text)
        except Exception as exc:
            notify("Earshot — error", str(exc))
        finally:
            self.transcribing = False
            self._set_icon_color(COLOR_IDLE)
            if self.icon is not None:
                self.icon.update_menu()

    # -- menu actions --------------------------------------------------

    def _toggle_listening(self, icon: Icon, _item: MenuItem) -> None:
        self.listening = not self.listening
        if self.listening:
            self.recorder.start_stream()
        else:
            self.recorder.stop_stream()
            if self.recording:
                self.recording = False
                self.recorder.stop()
            self.pressed.clear()
        self._set_icon_color(COLOR_IDLE)
        icon.update_menu()

    def _quit(self, icon: Icon, _item: MenuItem) -> None:
        self.listening = False
        if self.listener is not None:
            self.listener.stop()
        self.recorder.stop_stream()
        icon.stop()

    # -- lifecycle -----------------------------------------------------

    def run(self) -> None:
        self.recorder.start_stream()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

        icon = Icon(
            "earshot",
            icon=make_icon(COLOR_IDLE),
            title="Earshot",
            menu=Menu(
                MenuItem(
                    lambda i: f"Hold '{self.key_label}' to record",
                    None,
                    enabled=False,
                ),
                MenuItem(self._status_text, None, enabled=False),
                MenuItem(self._last_text_text, None, enabled=False),
                Menu.SEPARATOR,
                MenuItem(self._toggle_label, self._toggle_listening),
                Menu.SEPARATOR,
                MenuItem("Quit", self._quit),
            ),
        )
        self.icon = icon
        icon.run()


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System-tray speech-to-text application for Linux."
    )
    parser.add_argument("model", help="HuggingFace model id, e.g. openai/whisper-tiny")
    parser.add_argument(
        "--key",
        default="shift_r",
        help="Key (combination) to hold while recording (default: shift_r, "
        "the right Shift key). Combine keys with '+', e.g. 'ctrl+space', "
        "'shift_r+space', 'r'. Use 'shift_r'/'ctrl_r'/'alt_r' to require a "
        "specific right-side modifier; 'shift'/'ctrl'/'alt' match either side.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading '{args.model}' on {device} ({dtype})...")
    loaded = load_model(args.model, device=device, dtype=dtype)
    print("Model loaded. Tray icon started — close it from the menu to quit.")

    combo = parse_combo(args.key)
    app = TrayApp(
        loaded=loaded,
        combo=combo,
        key_label=pretty_label(args.key),
        typer=make_typer(),
    )
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
