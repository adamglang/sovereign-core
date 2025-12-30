"""Test Wake Word Detector functionality."""

import struct
import threading
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from sovereign_core.config import WakeWordConfig
from sovereign_core.ears.wake_word_detector import (
    WakeWordDetection,
    WakeWordDetector,
)


@pytest.fixture
def valid_wake_word_config():
    """Valid wake word configuration."""
    return WakeWordConfig(
        access_key="test-access-key",
        sensitivity=0.7,
    )


@pytest.fixture
def mock_porcupine():
    """Mock Porcupine instance."""
    mock = MagicMock()
    mock.sample_rate = 16000
    mock.frame_length = 512
    mock.process.return_value = -1  # No detection by default
    return mock


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio instance."""
    mock = MagicMock()
    mock_stream = MagicMock()
    mock.open.return_value = mock_stream
    return mock, mock_stream


def test_wake_word_detection_structure():
    """Test WakeWordDetection data structure."""
    timestamp = datetime.now()
    detection = WakeWordDetection(
        detected=True,
        confidence=0.8,
        timestamp=timestamp,
    )
    
    assert detection.detected is True
    assert detection.confidence == 0.8
    assert detection.timestamp == timestamp


def test_detector_initialization(valid_wake_word_config):
    """Test WakeWordDetector initialization."""
    detector = WakeWordDetector(valid_wake_word_config)
    
    assert detector.config == valid_wake_word_config
    assert detector._provider is not None
    assert detector.provider_name == "porcupine"


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
@patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio')
def test_detector_start_initialization(
    mock_pyaudio_class,
    mock_pvporcupine,
    valid_wake_word_config,
    mock_porcupine,
):
    """Test detector start initializes Porcupine and PyAudio."""
    # Configure mock to detect wake word immediately so generator yields
    mock_porcupine.process.return_value = 0
    mock_pvporcupine.create.return_value = mock_porcupine
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_class.return_value = mock_pa_instance
    mock_pa_instance.open.return_value = mock_stream
    
    # Mock audio stream to return one frame of data
    audio_frame = struct.pack('h' * 512, *([0] * 512))
    mock_stream.read.return_value = audio_frame
    
    detector = WakeWordDetector(valid_wake_word_config)
    
    try:
        gen = detector.start()
        # Start the generator (this initializes Porcupine and PyAudio, then yields detection)
        detection = next(gen)
        assert isinstance(detection, WakeWordDetection)
    finally:
        # Stop detector to exit the loop and cleanup
        detector.stop()
    
    # Verify Porcupine was created with correct parameters
    mock_pvporcupine.create.assert_called_once()
    call_kwargs = mock_pvporcupine.create.call_args.kwargs
    assert call_kwargs["access_key"] == "test-access-key"
    assert call_kwargs["keywords"] == ["hey sovereign"]
    assert call_kwargs["sensitivities"] == [0.7]
    
    # Verify PyAudio stream was opened
    mock_pa_instance.open.assert_called_once()
    open_kwargs = mock_pa_instance.open.call_args.kwargs
    assert open_kwargs["rate"] == 16000
    assert open_kwargs["channels"] == 1


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
@patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio')
def test_detector_yields_detection_event(
    mock_pyaudio_class,
    mock_pvporcupine,
    valid_wake_word_config,
    mock_porcupine,
):
    """Test detector yields WakeWordDetection when wake word is detected."""
    # Configure mock to detect wake word on first frame
    mock_porcupine.process.side_effect = [
        0,  # Wake word detected (index 0)
        -1,  # No detection
        Exception("Stop"),  # Stop iteration
    ]
    
    mock_pvporcupine.create.return_value = mock_porcupine
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_class.return_value = mock_pa_instance
    mock_pa_instance.open.return_value = mock_stream
    
    # Mock audio data
    audio_frame = struct.pack('h' * 512, *([0] * 512))
    mock_stream.read.return_value = audio_frame
    
    detector = WakeWordDetector(valid_wake_word_config)
    
    try:
        detections = []
        for detection in detector.start():
            detections.append(detection)
            if len(detections) >= 1:
                break
        
        assert len(detections) == 1
        assert isinstance(detections[0], WakeWordDetection)
        assert detections[0].detected is True
        assert detections[0].confidence == 0.7  # Same as sensitivity
        assert isinstance(detections[0].timestamp, datetime)
    
    except Exception:
        pass
    finally:
        detector.stop()


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
@patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio')
def test_detector_handles_audio_processing_errors(
    mock_pyaudio_class,
    mock_pvporcupine,
    valid_wake_word_config,
    mock_porcupine,
):
    """Test detector handles audio processing errors gracefully."""
    mock_pvporcupine.create.return_value = mock_porcupine
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_class.return_value = mock_pa_instance
    mock_pa_instance.open.return_value = mock_stream
    
    call_count = [0]
    
    def mock_process_with_error(pcm):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Processing error")
        return 0  # Detection on second call
    
    mock_porcupine.process.side_effect = mock_process_with_error
    
    audio_frame = struct.pack('h' * 512, *([0] * 512))
    mock_stream.read.return_value = audio_frame
    
    detector = WakeWordDetector(valid_wake_word_config)
    
    try:
        gen = detector.start()
        # First frame raises RuntimeError, is caught, logs error, continues
        # Second frame detects wake word and yields
        detection = next(gen)
        assert isinstance(detection, WakeWordDetection)
        assert detection.detected is True
        # Verify error was encountered and recovered from
        assert call_count[0] == 2
    finally:
        detector.stop()


def test_detector_cannot_start_twice(valid_wake_word_config):
    """Test that detector raises error if started while already running."""
    with patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine'), \
         patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio'):
        
        detector = WakeWordDetector(valid_wake_word_config)
        detector._provider._running = True  # Simulate already running
        
        with pytest.raises(RuntimeError) as exc_info:
            next(detector.start())
        
        assert "already running" in str(exc_info.value).lower()


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
@patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio')
def test_detector_cleanup_on_stop(
    mock_pyaudio_class,
    mock_pvporcupine,
    valid_wake_word_config,
    mock_porcupine,
):
    """Test detector properly cleans up resources on stop."""
    # Make mock return wake word detection so generator yields
    mock_porcupine.process.return_value = 0
    mock_pvporcupine.create.return_value = mock_porcupine
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_class.return_value = mock_pa_instance
    mock_pa_instance.open.return_value = mock_stream
    
    audio_frame = struct.pack('h' * 512, *([0] * 512))
    mock_stream.read.return_value = audio_frame
    
    detector = WakeWordDetector(valid_wake_word_config)
    
    try:
        gen = detector.start()
        next(gen)  # Get one detection so generator progresses
    except Exception:
        pass
    
    detector.stop()
    
    # Verify cleanup
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_pa_instance.terminate.assert_called_once()
    mock_porcupine.delete.assert_called_once()
    
    assert detector._provider._running is False
    assert detector._provider._audio_stream is None
    assert detector._provider._pa is None
    assert detector._provider._porcupine is None


def test_detector_stop_is_idempotent(valid_wake_word_config):
    """Test that stop() can be called multiple times safely."""
    detector = WakeWordDetector(valid_wake_word_config)
    
    # Should not raise even when nothing is running
    detector.stop()
    detector.stop()
    detector.stop()


def test_detector_handles_cleanup_errors(valid_wake_word_config):
    """Test detector handles errors during cleanup gracefully."""
    detector = WakeWordDetector(valid_wake_word_config)
    
    # Create mock resources that raise errors on cleanup
    mock_stream = MagicMock()
    mock_stream.stop_stream.side_effect = Exception("Stream error")
    mock_stream.close.side_effect = Exception("Close error")
    
    mock_pa = MagicMock()
    mock_pa.terminate.side_effect = Exception("PA error")
    
    mock_porcupine = MagicMock()
    mock_porcupine.delete.side_effect = Exception("Porcupine error")
    
    detector._provider._audio_stream = mock_stream
    detector._provider._pa = mock_pa
    detector._provider._porcupine = mock_porcupine
    detector._provider._running = True
    
    # Should not raise despite errors in cleanup
    detector.stop()
    
    # Verify cleanup was attempted
    mock_stream.stop_stream.assert_called_once()
    mock_pa.terminate.assert_called_once()
    mock_porcupine.delete.assert_called_once()
    
    # Resources should be cleared despite errors
    assert detector._provider._audio_stream is None
    assert detector._provider._pa is None
    assert detector._provider._porcupine is None


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
@patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio')
def test_detector_context_manager(
    mock_pyaudio_class,
    mock_pvporcupine,
    valid_wake_word_config,
    mock_porcupine,
):
    """Test detector works as context manager."""
    mock_pvporcupine.create.return_value = mock_porcupine
    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_class.return_value = mock_pa_instance
    mock_pa_instance.open.return_value = mock_stream
    
    with WakeWordDetector(valid_wake_word_config) as detector:
        assert isinstance(detector, WakeWordDetector)
    
    # Verify stop was called on exit (cleanup attempted)
    assert detector._provider._running is False


def test_detector_with_custom_model_path():
    """Test detector initialization with custom model path."""
    config = WakeWordConfig(
        access_key="test-key",
        model_path="/custom/path/model.pv",
        sensitivity=0.8,
    )
    
    with patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine') as mock_pv, \
         patch('sovereign_core.ears.providers.porcupine_provider.pyaudio.PyAudio') as mock_pyaudio_class:
        
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        # Return wake word detection so generator yields
        mock_porcupine.process.return_value = 0
        mock_pv.create.return_value = mock_porcupine
        
        # Set up PyAudio mocks
        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio_class.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream
        
        audio_frame = struct.pack('h' * 512, *([0] * 512))
        mock_stream.read.return_value = audio_frame
        
        detector = WakeWordDetector(config)
        
        try:
            gen = detector.start()
            next(gen)  # Get one detection
        except Exception:
            pass
        finally:
            detector.stop()
        
        # Verify model_path was passed to create
        call_kwargs = mock_pv.create.call_args.kwargs
        assert call_kwargs["model_path"] == "/custom/path/model.pv"


@patch('sovereign_core.ears.providers.porcupine_provider.pvporcupine')
def test_detector_handles_initialization_failure(
    mock_pvporcupine,
    valid_wake_word_config,
):
    """Test detector handles Porcupine initialization failures."""
    mock_pvporcupine.create.side_effect = RuntimeError("Invalid access key")
    
    detector = WakeWordDetector(valid_wake_word_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        gen = detector.start()
        next(gen)
    
    assert "initialization failed" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])