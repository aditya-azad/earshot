# earshot

Speech-to-text experiments using HuggingFace transformers.

## Testing a model from the terminal

`scripts/test_stt.py` lets you hold a key to record speech from your microphone
and prints the model's transcription to the terminal.

### Usage

```sh
uv run scripts/test_stt.py <model-id>
```

Examples:

```sh
uv run scripts/test_stt.py openai/whisper-tiny
uv run scripts/test_stt.py openai/whisper-large-v3
uv run scripts/test_stt.py facebook/wav2vec2-base-960h
```

### Controls

- **Hold space** to record; **release** to transcribe.
- **Esc** to quit.

Run `uv run scripts/test_stt.py --help` for available options.

### Notes

- On Linux, `pynput` requires an active display session (X11 or a Wayland
  session that supports global input events) to capture key presses.

## Linux system-tray application

`scripts/linux.py` runs as a background system-tray app. It types the
transcription directly at your cursor using `xdotool` (X11) or `wtype`
(Wayland).

### Usage

```sh
uv run scripts/linux.py <model-id>
uv run scripts/linux.py openai/whisper-tiny --key shift_r
uv run scripts/linux.py openai/whisper-tiny --key ctrl+space
```

### Controls

- **Hold the trigger key** to record; **release** to transcribe and type.
- Quit from the tray menu.

Run `uv run scripts/linux.py --help` for available options.

### Requirements

- GNOME `AppIndicator` extension (or equivalent KStatusNotifierItem support)
- `notify-send` for desktop notifications
- `xdotool` (X11) or `wtype` (Wayland) for typing
