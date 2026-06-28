"""Display-server-specific text typing (strategy pattern).

Typing transcribed text at the cursor requires a different external tool
depending on the display server in use: ``xdotool`` on X11 and ``wtype``
on Wayland.  :class:`TextTyper` defines the common interface; the
concrete subclasses implement it for each display server, and
:func:`make_typer` picks the right one based on ``XDG_SESSION_TYPE``.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod


class TextTyper(ABC):
    """Abstract interface for typing text at the current cursor position."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type *text* at the cursor, silently ignoring missing tools."""
        raise NotImplementedError()


class X11Typer(TextTyper):
    """Types text using ``xdotool`` (X11)."""

    def type_text(self, text: str) -> None:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", text],
            check=False,
            capture_output=True,
        )


class WaylandTyper(TextTyper):
    """Types text using ``wtype`` (Wayland)."""

    def type_text(self, text: str) -> None:
        subprocess.run(
            ["wtype", text],
            check=False,
            capture_output=True,
        )


def make_typer(session_type: str | None = None) -> TextTyper:
    """Return a :class:`TextTyper` for the current display server.

    *session_type* defaults to the ``XDG_SESSION_TYPE`` environment
    variable.  Wayland sessions get a :class:`WaylandTyper`; everything
    else (X11 and unknown) falls back to :class:`X11Typer`.
    """
    if session_type is None:
        session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        return WaylandTyper()
    return X11Typer()
