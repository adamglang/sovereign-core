# Sovereign Core - Package Structure

This directory contains the main `sovereign_core` Python package for the voice assistant POC.

## Package Organization

```
src/sovereign_core/
├── __init__.py          # Package initialization
├── ears/                # Wake-word detection and audio capture
├── brain/               # Speech-to-text and LLM processing
├── router/              # Conversation vs action routing logic
├── mouth/               # Text-to-speech output
└── ipc/                 # SQLite-based inter-process communication
```

## Module Responsibilities

### `ears/`
- **Wake-word Detection**: Uses Picovoice Porcupine for local wake-word detection
- **Audio Capture**: Captures audio input using sounddevice
- Listens continuously for the wake word, then captures user speech

### `brain/`
- **Speech-to-Text (STT)**: Uses faster-whisper for local transcription (GPU-capable)
- **LLM Processing**: OpenAI API integration with swappable backend design
- Converts speech to text and processes it through the language model

### `router/`
- **Intent Classification**: Determines whether input is conversation or action request
- **Message Routing**: Routes to appropriate handler (conversation loop or executor)
- Acts as the decision-making layer between understanding and action

### `mouth/`
- **Text-to-Speech (TTS)**: Windows native TTS initially (Piper TTS evaluation pending)
- **Audio Output**: Speaks responses back to the user
- Converts LLM responses to natural-sounding speech

### `ipc/`
- **SQLite Communication**: Lightweight IPC between sovereign-core and sovereign-executor
- **Message Queue**: Manages action requests and responses
- Provides async communication without complex networking

## Configuration

Configuration is managed through `config.yaml` in the project root. Key settings include:

- Wake-word sensitivity and model paths
- Audio device configuration
- STT model size and device (CPU/GPU)
- LLM provider and API settings
- TTS provider and voice settings
- SQLite database path for IPC

**Security Note**: Sensitive values like API keys should be set via environment variables (e.g., `OPENAI_API_KEY`, `PORCUPINE_ACCESS_KEY`).

## Installation

From the project root (`sovereign-core/`):

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
# Linting and formatting
ruff check .
ruff format .

# Type checking
mypy src/
```

## Technology Stack

- **Wake-word**: Picovoice Porcupine
- **Audio I/O**: sounddevice
- **STT**: faster-whisper (local, GPU-capable)
- **LLM**: OpenAI API (swappable backend)
- **TTS**: Windows SAPI (native)
- **IPC**: SQLite
- **Validation**: Pydantic

## Next Steps

This is a POC structure. Implementation order:

1. **IPC Layer**: Set up SQLite communication
2. **Ears**: Wake-word detection and audio capture
3. **Brain**: STT integration and LLM abstraction
4. **Mouth**: Windows native TTS
5. **Router**: Intent classification and routing logic

See `docs/architecture.md` for detailed architectural decisions and rationale.