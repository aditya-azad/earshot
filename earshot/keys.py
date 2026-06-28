from __future__ import annotations

from pynput import keyboard

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

_MODIFIER_PAIRS = [
    (keyboard.Key.shift, keyboard.Key.shift_r),
    (keyboard.Key.ctrl, keyboard.Key.ctrl_r),
    (keyboard.Key.alt, keyboard.Key.alt_r),
    (keyboard.Key.cmd, keyboard.Key.cmd_r),
]

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


def normalize_key(key, combo: set) -> object:
    """Normalize a pynput key for comparison against *combo*.

    Like pynput's Listener.canonical, but keeps left/right modifiers
    distinct when the combo pins a specific side (e.g. shift_r), so
    only that side matches.  Generic modifier names collapse to the
    left form and match either side.
    """
    if isinstance(key, keyboard.KeyCode):
        if key.char is not None:
            return keyboard.KeyCode.from_char(key.char.lower())
        return key
    if isinstance(key, keyboard.Key):
        for left, right in _MODIFIER_PAIRS:
            if key in (left, right):
                return key if right in combo else left
        if key in _MODIFIERS.values():
            return key
        if key.value.vk is not None:
            return keyboard.KeyCode.from_vk(key.value.vk)
    return key
