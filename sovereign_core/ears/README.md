# Ears Module - Wake Word Detection and Audio Capture

This module provides pluggable wake word detection, VAD-based turn-taking, and audio capture capabilities for Sovereign Core.

## Architecture

The module follows a provider pattern for wake word detection, allowing easy swapping of different wake word engines without changing application code.

### Components

#### Wake Word Detection

- **[`wake_word_interface.py`](wake_word_interface.py)** - Abstract base class defining the wake word provider interface
- **[`wake_word_factory.py`](wake_word_factory.py)** - Factory for creating wake word provider instances
- **[`wake_word_detector.py`](wake_word_detector.py)** - Backward-compatible wrapper maintaining the original API
- **[`providers/`](providers/)** - Concrete wake word engine implementations
  - **[`porcupine_provider.py`](providers/porcupine_provider.py)** - Picovoice Porcupine implementation

#### Audio Capture and Turn-Taking

- **[`audio_capture.py`](audio_capture.py)** - Audio recording with VAD-based turn-taking
  - Intelligent endpointing using WebRTC VAD
  - State machine for natural conversation flow
  - Configurable grace periods and silence thresholds
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

### Audio Capture with VAD-Based Turn-Taking

```python
from sovereign_core.ears import AudioCapture
from sovereign_core.config import AudioConfig, TurnTakingConfig

audio_config = AudioConfig(
    sample_rate=16000,
    channels=1
)

turn_taking_config = TurnTakingConfig(
    vad_aggressiveness=2,              # 0-3 sensitivity
    end_silence_duration_ms=700,       # Silence to end utterance
    post_speech_grace_ms=500,          # Grace period for trailing sounds
    max_recording_duration_s=15        # Safety ceiling
)

capture = AudioCapture(
    config=audio_config,
    turn_taking_config=turn_taking_config
)

# Capture with intelligent endpointing
recording = capture.capture_with_vad()
print(f"Captured {len(recording.audio_data)} samples")

# Or use fixed duration (backward compatible)
recording = capture.capture(duration=5.0)
```

### Turn-Taking Configuration

The VAD-based turn-taking system uses a 5-state machine:
1. **WAITING_FOR_SPEECH** - Buffering until speech detected
2. **SPEECH_DETECTED** - Actively recording
3. **SILENCE_AFTER_SPEECH** - Silence detected, accumulating duration
4. **GRACE_PERIOD** - Silence threshold met, cancellable if speech resumes
5. **FINALIZED** - Recording complete

**Tuning Parameters** (in `config.yaml`):
- `vad_aggressiveness` (0-3): Higher = more aggressive silence detection
- `min_speech_duration_ms`: Minimum speech duration to process (filters noise)
- `end_silence_duration_ms`: Silence duration to consider utterance ended
- `post_speech_grace_ms`: Grace period after silence (captures trailing sounds)
- `max_recording_duration_s`: Safety ceiling to prevent infinite recording
- `vad_frame_duration_ms`: VAD processing frame size (10/20/30ms)

**Common Adjustments:**
- If cut off mid-sentence: Increase `end_silence_duration_ms` and `post_speech_grace_ms`
- If waits too long: Decrease `end_silence_duration_ms` and `post_speech_grace_ms`
- If background noise triggers: Increase `vad_aggressiveness` and `min_speech_duration_ms`

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

Wake word detection and turn-taking are configured via [`config.yaml`](../../config.yaml):

```yaml
wake_word:
  provider: porcupine  # Can be changed to any supported provider
  access_key: ${PORCUPINE_ACCESS_KEY}
  sensitivity: 0.7
  model_path: null  # Optional custom model

turn_taking:
  vad_aggressiveness: 2              # 0-3, higher = more aggressive
  min_speech_duration_ms: 300        # Filters false starts
  end_silence_duration_ms: 700       # Natural pause detection
  post_speech_grace_ms: 500          # Capture trailing sounds
  max_recording_duration_s: 15       # Safety ceiling
  vad_frame_duration_ms: 30          # Optimal for 16kHz audio
```

## Testing

Tests are located in:
- [`../../tests/test_wake_word_detector.py`](../../tests/test_wake_word_detector.py) - Wake word detection
- [`../../tests/test_audio_capture.py`](../../tests/test_audio_capture.py) - Audio capture
- [`../../tests/test_turn_taking.py`](../../tests/test_turn_taking.py) - VAD turn-taking

Run tests with:
```bash
pytest tests/test_wake_word_detector.py -v
pytest tests/test_audio_capture.py -v
pytest tests/test_turn_taking.py -v
```

## Design Principles

1. **Provider Pattern**: Easy swapping of wake word engines without code changes
2. **Direct Implementation**: VAD uses single implementation (WebRTC) - no provider abstraction (matches TTS pattern)
3. **Backward Compatibility**: Existing code continues to work through the wrapper
4. **Clean Abstractions**: Providers implement a simple, focused interface
5. **Factory Creation**: Centralized provider instantiation for consistency
6. **Configuration-Driven**: Provider selection and behavior via config, not code
7. **Structured Logging**: All turn-taking events logged with timing for debugging

## Architecture Decisions

### Why Direct Implementation for VAD?

Unlike wake word detection (which uses a provider pattern), VAD turn-taking uses **direct implementation** following the TTS precedent:

- **Not user-facing choice**: VAD is internal implementation detail, not a feature users select
- **Single correct solution**: WebRTC VAD is industry standard, no compelling alternatives
- **Tight coupling**: VAD is intrinsic to audio capture, not a separate concern
- **YAGNI principle**: No current need for VAD swapping; premature abstraction adds complexity

See [`docs/major_work_completed/turn-taking-vad-implementation.md`](../../docs/major_work_completed/turn-taking-vad-implementation.md) for full implementation details.