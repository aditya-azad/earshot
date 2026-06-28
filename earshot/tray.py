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
    earshot openai/whisper-tiny
    earshot openai/whisper-tiny --key shift_r
    earshot openai/whisper-tiny --key ctrl+space
    earshot                  # uses $EARSHOT_MODEL or openai/whisper-large-v3
"""

from __future__ import annotations

import os
import threading

import numpy as np
from pynput import keyboard
from pystray import Icon, Menu, MenuItem

from .audio import Recorder
from . import backend  # noqa: F401 – import side-effect: selects pystray backend
from .cli import build_parser, load_for_cli, run_entry_point
from .config import COLOR_IDLE, COLOR_RECORDING, COLOR_TRANSCRIBING, DEFAULT_KEY
from .icon import make_icon
from .keys import normalize_key, parse_combo, pretty_label
from .model import LoadedModel
from .notify import notify
from .text_input import TextTyper, make_typer
from .transcribe import transcribe


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
        self._menu_visible = False

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

    def _last_text_label(self, _icon: Icon) -> str:
        if self.last_text:
            truncated = self.last_text[:50]
            if len(self.last_text) > 50:
                truncated += "…"
            return f"Last: {truncated}"
        return "No transcription yet"

    def _toggle_label(self, _icon: Icon) -> str:
        return "Pause listening" if self.listening else "Resume listening"

    # -- keyboard callbacks --------------------------------------------

    def _on_press(self, key) -> None:
        if not self.listening or self.transcribing:
            return
        self.pressed.add(normalize_key(key, self.combo))
        if not self.recording and self.combo.issubset(self.pressed):
            self.recording = True
            self.recorder.start()
            self._set_icon_color(COLOR_RECORDING)

    def _on_release(self, key) -> None:
        self.pressed.discard(normalize_key(key, self.combo))
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
        except Exception as exc:
            notify("Earshot — error", str(exc))
        finally:
            self.transcribing = False
            self._set_icon_color(COLOR_IDLE)
            if self.icon is not None:
                self.icon.update_menu()

    # -- menu actions --------------------------------------------------

    def _toggle_listening(self, icon: Icon, _item: MenuItem | None) -> None:
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

    def _quit(self, icon: Icon, _item: MenuItem | None) -> None:
        self.listening = False
        if self.listener is not None:
            self.listener.stop()
        self.recorder.stop_stream()
        icon.stop()

    def _show_menu(self, icon: Icon, _item: MenuItem) -> None:
        if self._menu_visible:
            return
        si = getattr(icon, "_status_icon", None)
        if si is not None:
            from gi.repository import Gtk  # type: ignore[missing-module-attribute]

            mh = getattr(icon, "_menu_handle", None)
            if mh is not None:
                mh.show_all()
                mh.popup(
                    None,
                    None,
                    Gtk.StatusIcon.position_menu,
                    si,
                    1,
                    Gtk.get_current_event_time(),
                )
        elif not hasattr(icon, "_appindicator"):
            self._menu_visible = True
            threading.Thread(
                target=self._zenity_popup_menu, args=(icon,), daemon=True
            ).start()

    def _zenity_popup_menu(self, icon: Icon) -> None:
        try:
            import subprocess

            toggle = self._toggle_label(icon)
            status = self._status_text(icon)
            last = self._last_text_label(icon)
            title = f"Earshot — {status} | {last}"
            proc = subprocess.run(
                [
                    "zenity",
                    "--list",
                    "--column=Action",
                    "--hide-header",
                    f"--title={title}",
                    toggle,
                    "Quit",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            choice = proc.stdout.strip()
            if choice == "Quit":
                self._quit(icon, None)
            elif choice == toggle:
                self._toggle_listening(icon, None)
        except Exception:
            pass
        finally:
            self._menu_visible = False

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
                    "Show Menu",
                    self._show_menu,
                    default=True,
                    visible=False,
                ),
                MenuItem(
                    lambda i: f"Hold '{self.key_label}' to record",
                    None,
                    enabled=False,
                ),
                MenuItem(self._status_text, None, enabled=False),
                MenuItem(self._last_text_label, None, enabled=False),
                Menu.SEPARATOR,
                MenuItem(self._toggle_label, self._toggle_listening),
                Menu.SEPARATOR,
                MenuItem("Quit", self._quit),
            ),
        )
        self.icon = icon

        if not hasattr(icon, "_status_icon") and not hasattr(icon, "_appindicator"):
            try:
                import Xlib.X as X

                def _on_click(event):
                    if event.detail in (1, 3):
                        icon()

                icon._message_handlers[X.ButtonPress] = _on_click
            except ImportError:
                pass

        icon.run()


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    default_key = os.environ.get("EARSHOT_KEY", DEFAULT_KEY)
    key_help = (
        f"Key (combination) to hold while recording (default: {default_key}, "
        "or $EARSHOT_KEY if set). Combine keys with '+', e.g. 'ctrl+space', "
        "'shift_r+space', 'r'. Use 'shift_r'/'ctrl_r'/'alt_r' to require a "
        "specific right-side modifier; 'shift'/'ctrl'/'alt' match either side."
    )
    parser = build_parser(
        "System-tray speech-to-text application for Linux.",
        key_default=default_key,
        key_help=key_help,
    )
    args = parser.parse_args()

    loaded = load_for_cli(args)
    print("Tray icon started — close it from the menu to quit.")

    combo = parse_combo(args.key)
    app = TrayApp(
        loaded=loaded,
        combo=combo,
        key_label=pretty_label(args.key),
        typer=make_typer(),
    )
    app.run()


if __name__ == "__main__":
    run_entry_point(main)
