from __future__ import annotations

import os


def select_pystray_backend() -> None:
    """Set ``PYSTRAY_BACKEND`` based on available typelibs.

    pystray's backend selector catches ImportError but not the ValueError
    that gi.require_version raises when an AppIndicator3 typelib is absent.
    With PyGObject installed but the typelib missing, importing pystray
    would crash.  Select the gtk backend (native tray menu, needs only
    Gtk-3.0) when appindicator is unavailable, or xorg when gi/Gtk itself
    is missing — so the app always starts instead of crashing.

    Must be called before importing pystray.
    """
    if "PYSTRAY_BACKEND" in os.environ:
        return
    backend = None
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        try:
            gi.require_version("AppIndicator3", "0.1")
        except ValueError:
            try:
                gi.require_version("AyatanaAppIndicator3", "0.1")
            except ValueError:
                backend = "gtk"
    except (ImportError, ValueError):
        backend = "xorg"
    if backend:
        os.environ["PYSTRAY_BACKEND"] = backend


select_pystray_backend()
