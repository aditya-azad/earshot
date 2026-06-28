"""Linux desktop notification helper.

Sends notifications via ``notify-send``, which works on both X11 and
Wayland.  Failures (e.g. ``notify-send`` not installed) are silently
ignored so the application keeps working without a notification daemon.
"""

from __future__ import annotations

import subprocess


def notify(title: str, body: str = "") -> None:
    try:
        subprocess.run(
            ["notify-send", "--app-name=Earshot", title, body],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass
