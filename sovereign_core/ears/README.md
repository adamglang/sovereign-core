# Ears Module - Wake Word Detection and Audio Capture

This module provides pluggable wake word detection and audio capture capabilities for Sovereign Core.

## Architecture

The module follows a provider pattern for wake word detection, allowing easy swapping of different wake word engines without changing application code.

### Components

#### Wake Word Detection

- **[`wake_word_interface.py`](wake_word_interface.py)** - Abstract base class defining the wake word provider interface
- **[`wake_word_factory.py`](wake_word_factory.py)** - Factory for creating wake word provider instances
- **[`wake_word_detector.py`](wake_word_detector.py)** - Backward-compatible wrapper maintaining the original API
- **[`providers/`](providers/)** - Concrete wake word engine implementations
  - **[`porcupine_provider.py`](providers/porcupine_provider.py)** - Picovoice Porcupine implementation

#### Audio Capture

- **[`audio_capture.py`](audio_capture.py)** - Audio recording after wake word detection
- **[`speech_to_text.py`](speech_to_text.py)** - Speech transcription using Whisper

## Usage

### Basic Wake Word Detection

```python
from sovereign_core.ears import get_wake_word_provider

# Create a wake word provider
provider = get_wake_word_provider(
    provider_name="porcupine",
    config={
        "access_key": "your-porcupine-key",
        "keywords": ["hey sovereign"],
        "sensitivity": 0.7
    }
)

# Listen for wake word
for detection in provider.start():
    print(f"Wake word detected at {detection.timestamp}")
    print(f"Confidence: {detection.confidence}")
```

### Using the Detector Wrapper (Backward Compatible)

```python
from sovereign_core.ears import WakeWordDetector
from sovereign_core.config import WakeWordConfig

config = WakeWordConfig(
    access_key="your-key",
    sensitivity=0.7
)

detector = WakeWordDetector(config)

# Wait for single detection
if detector.wait_for_wake_word():
    print("Wake word detected!")

detector.stop()
```

## Adding New Wake Word Providers

To add support for a new wake word engine:

1. **Create a provider class** in `providers/` that inherits from [`WakeWordProvider`](wake_word_interface.py:22)
2. **Implement required methods**: [`start()`](wake_word_interface.py:40) and [`stop()`](wake_word_interface.py:57)
3. **Export the provider** in [`providers/__init__.py`](providers/__init__.py)
4. **Register in factory** by adding a case in [`wake_word_factory.py`](wake_word_factory.py:20)

### Example Provider Template

```python
from ..wake_word_interface import WakeWordDetection, WakeWordProvider

class CustomProvider(WakeWordProvider):
    def __init__(self, **config):
        # Initialize your wake word engine
        pass
    
    def start(self) -> Iterator[WakeWordDetection]:
        # Start listening and yield detections
        while self._running:
            # Process audio...
            if wake_word_detected:
                yield WakeWordDetection(
                    detected=True,
                    confidence=0.8,
                    timestamp=datetime.now()
                )
    
    def stop(self) -> None:
        # Clean up resources
        self._running = False
```

## Supported Providers

### Porcupine (Default)

- **Type**: Offline, low-CPU wake word detection
- **Vendor**: Picovoice
- **Requirements**: Porcupine access key from https://console.picovoice.ai/
- **Configuration**:
  - `access_key` (required): Picovoice API key
  - `keywords` (optional): List of wake words (default: ["hey sovereign"])
  - `sensitivity` (optional): 0.0-1.0 (default: 0.5)
  - `model_path` (optional): Path to custom model file

### Future Providers

- **Snowboy** - Custom wake word training
- **Mycroft Precise** - Open-source wake word detection
- **Pocketsphinx** - CMU Sphinx-based detection
- **Custom** - Your own implementation

## Configuration

Wake word detection is configured via [`config.yaml`](../../config.yaml):

```yaml
wake_word:
  provider: porcupine  # Can be changed to any supported provider
  access_key: ${PORCUPINE_ACCESS_KEY}
  sensitivity: 0.7
  model_path: null  # Optional custom model
```

## Testing

Tests are located in [`../../tests/test_wake_word_detector.py`](../../tests/test_wake_word_detector.py).

Run tests with:
```bash
pytest tests/test_wake_word_detector.py -v
```

## Design Principles

1. **Provider Pattern**: Easy swapping of wake word engines without code changes
2. **Backward Compatibility**: Existing code continues to work through the wrapper
3. **Clean Abstractions**: Providers implement a simple, focused interface
4. **Factory Creation**: Centralized provider instantiation for consistency
5. **Configuration-Driven**: Provider selection via config, not code