# Earshot

Speech-to-text application for Linux using HuggingFace transformers.

## OS support

- [x] Linux
- [ ] MacOS
- [ ] Windows

## Installation

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick start (editable)

```sh
uv sync
```

This installs all dependencies and the `earshot` package in **editable
mode** — edits to the source take effect the next time you relaunch,
with no reinstall needed.

### Install (global command + desktop integration)

A single script installs the global `earshot` command (editable, via
`uv tool`) and the desktop integration (launcher entry + icon +
wrapper):

```sh
scripts/install_linux.sh
```

The installed launcher runs the editable source from this project
directory, so editing the code and relaunching picks up your changes
immediately.

To remove everything:

```sh
scripts/install_linux.sh --uninstall
```

## Usage

### System-tray application

```sh
uv run earshot
uv run earshot openai/whisper-tiny
uv run earshot openai/whisper-tiny --key shift_r
uv run earshot openai/whisper-tiny --key ctrl+space
```

The model defaults to `openai/whisper-tiny` (or `$EARSHOT_MODEL` if
set).  The trigger key defaults to `shift_r` (or `$EARSHOT_KEY` if set).

### Terminal tester

```sh
uv run test-stt
uv run test-stt openai/whisper-tiny
uv run test-stt openai/whisper-large-v3
uv run test-stt facebook/wav2vec2-base-960h
```

### Controls

- **Tray app**: hold the trigger key to record; release to transcribe and
  type.  Quit from the tray menu.
- **Terminal tester**: hold space to record; release to transcribe.
  Press Esc to quit.

Run `earshot --help` or `test-stt --help` for all options.

## Requirements

- GNOME `AppIndicator` extension (or equivalent KStatusNotifierItem support)
- `notify-send` for desktop notifications
- `xdotool` (X11) or `wtype` (Wayland) for typing
- On Linux, `pynput` requires an active display session (X11 or a Wayland
  session that supports global input events) to capture key presses.
