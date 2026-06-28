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

### Options


| Flag    | Default | Description                                               |
| ------- | ------- | --------------------------------------------------------- |
| `--key` | `space` | Key to hold while recording (e.g. `r`, `space`, `enter`). |


### Notes

- On Linux, `pynput` requires an active display session (X11 or a Wayland
  session that supports global input events) to capture key presses.
