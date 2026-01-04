"""Test Audio Capture functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sovereign_core.config import AudioConfig, TurnTakingConfig
from sovereign_core.ears.audio_capture import AudioCapture, AudioRecording


@pytest.fixture
def valid_audio_config():
    """Valid audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        device_index=None,
    )


@pytest.fixture
def stereo_audio_config():
    """Stereo audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=2,
        device_index=0,
    )


@pytest.fixture
def turn_taking_config():
    """Turn-taking configuration with default values."""
    return TurnTakingConfig(
        vad_aggressiveness=2,
        min_speech_duration_ms=300,
        end_silence_duration_ms=700,
        post_speech_grace_ms=500,
        max_recording_duration_s=15,
        vad_frame_duration_ms=30,
    )


def test_audio_recording_structure():
    """Test AudioRecording data structure."""
    audio_data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    recording = AudioRecording(audio_data=audio_data, sample_rate=16000)
    
    assert isinstance(recording.audio_data, np.ndarray)
    assert recording.sample_rate == 16000
    np.testing.assert_array_equal(recording.audio_data, audio_data)


def test_audio_capture_initialization(valid_audio_config):
    """Test AudioCapture initialization."""
    capture = AudioCapture(valid_audio_config, default_duration=5.0)
    
    assert capture.config == valid_audio_config
    assert capture.default_duration == 5.0


def test_audio_capture_default_duration(valid_audio_config):
    """Test AudioCapture uses default duration when initialized."""
    capture = AudioCapture(valid_audio_config)
    assert capture.default_duration == 5.0


@patch('sovereign_core.ears.audio_capture.sd')
def test_successful_audio_capture(mock_sd, valid_audio_config):
    """Test successful audio capture with mocked sounddevice."""
    # Create mock audio data (mono)
    duration = 3.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=duration)
    recording = capture.capture()
    
    # Verify recording structure
    assert isinstance(recording, AudioRecording)
    assert isinstance(recording.audio_data, np.ndarray)
    assert recording.sample_rate == sample_rate
    assert len(recording.audio_data) == num_samples
    
    # Verify sounddevice was called correctly
    mock_sd.rec.assert_called_once_with(
        num_samples,
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        device=None,
    )
    mock_sd.wait.assert_called_once()


@patch('sovereign_core.ears.audio_capture.sd')
def test_capture_with_custom_duration(mock_sd, valid_audio_config):
    """Test audio capture with custom duration parameter."""
    duration = 2.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=5.0)
    recording = capture.capture(duration=duration)  # Override default
    
    assert len(recording.audio_data) == num_samples
    
    # Verify duration was used correctly
    call_args = mock_sd.rec.call_args
    assert call_args[0][0] == num_samples


@patch('sovereign_core.ears.audio_capture.sd')
def test_capture_uses_default_duration(mock_sd, valid_audio_config):
    """Test capture uses default duration when no duration specified."""
    default_duration = 4.0
    sample_rate = 16000
    num_samples = int(default_duration * sample_rate)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=default_duration)
    recording = capture.capture()  # No duration specified
    
    # Verify default duration was used
    call_args = mock_sd.rec.call_args
    assert call_args[0][0] == num_samples


@patch('sovereign_core.ears.audio_capture.sd')
def test_stereo_to_mono_conversion(mock_sd, stereo_audio_config):
    """Test stereo audio is converted to mono by taking first channel."""
    duration = 2.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Create stereo audio (2 channels)
    mock_stereo = np.random.randint(
        -32768, 32767,
        size=(num_samples, 2),
        dtype=np.int16
    )
    
    mock_sd.rec.return_value = mock_stereo
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(stereo_audio_config, default_duration=duration)
    recording = capture.capture()
    
    # Verify mono conversion (should take first channel)
    assert recording.audio_data.ndim == 1  # Should be 1D array
    assert len(recording.audio_data) == num_samples
    np.testing.assert_array_equal(recording.audio_data, mock_stereo[:, 0])


@patch('sovereign_core.ears.audio_capture.sd')
def test_mono_audio_flattened(mock_sd, valid_audio_config):
    """Test mono audio is properly flattened."""
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Create mono audio with shape (n, 1) to simulate sounddevice output
    mock_mono = np.random.randint(
        -32768, 32767,
        size=(num_samples, 1),
        dtype=np.int16
    )
    
    mock_sd.rec.return_value = mock_mono
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=duration)
    recording = capture.capture()
    
    # Should be flattened to 1D
    assert recording.audio_data.ndim == 1
    assert len(recording.audio_data) == num_samples


@patch('sovereign_core.ears.audio_capture.sd')
def test_microphone_access_failure(mock_sd, valid_audio_config):
    """Test handling of microphone access errors."""
    # Create a mock exception that looks like PortAudioError
    class MockPortAudioError(Exception):
        pass
    
    MockPortAudioError.__name__ = 'PortAudioError'
    mock_sd.rec.side_effect = MockPortAudioError("Device not found")
    
    capture = AudioCapture(valid_audio_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        capture.capture()
    
    assert "Failed to access microphone" in str(exc_info.value)


@patch('sovereign_core.ears.audio_capture.sd')
def test_general_capture_failure(mock_sd, valid_audio_config):
    """Test handling of general capture errors."""
    mock_sd.rec.side_effect = Exception("Unexpected error")
    
    capture = AudioCapture(valid_audio_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        capture.capture()
    
    assert "Audio capture failed" in str(exc_info.value)


@patch('sovereign_core.ears.audio_capture.sd')
def test_capture_with_device_index(mock_sd):
    """Test audio capture uses specified device index."""
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        device_index=2,  # Specific device
    )
    
    duration = 1.0
    num_samples = int(duration * config.sample_rate)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(config, default_duration=duration)
    capture.capture()
    
    # Verify device_index was passed
    call_kwargs = mock_sd.rec.call_args.kwargs
    assert call_kwargs["device"] == 2


@patch('sovereign_core.ears.audio_capture.sd')
def test_context_manager(mock_sd, valid_audio_config):
    """Test AudioCapture works as context manager."""
    mock_audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    with AudioCapture(valid_audio_config) as capture:
        assert isinstance(capture, AudioCapture)
        recording = capture.capture(duration=1.0)
        assert isinstance(recording, AudioRecording)


@patch('sovereign_core.ears.audio_capture.sd')
def test_multiple_captures(mock_sd, valid_audio_config):
    """Test multiple sequential captures work correctly."""
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Create different mock data for each capture
    mock_audio_1 = np.ones(num_samples, dtype=np.int16)
    mock_audio_2 = np.ones(num_samples, dtype=np.int16) * 2
    
    mock_sd.rec.side_effect = [mock_audio_1, mock_audio_2]
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=duration)
    
    recording_1 = capture.capture()
    recording_2 = capture.capture()
    
    # Verify both captures worked
    assert len(recording_1.audio_data) == num_samples
    assert len(recording_2.audio_data) == num_samples
    np.testing.assert_array_equal(recording_1.audio_data, mock_audio_1)
    np.testing.assert_array_equal(recording_2.audio_data, mock_audio_2)
    
    # Verify rec was called twice
    assert mock_sd.rec.call_count == 2


@patch('sovereign_core.ears.audio_capture.sd')
def test_audio_data_format_is_int16(mock_sd, valid_audio_config):
    """Test captured audio maintains int16 format."""
    duration = 1.0
    num_samples = int(duration * 16000)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    capture = AudioCapture(valid_audio_config, default_duration=duration)
    recording = capture.capture()
    
    assert recording.audio_data.dtype == np.int16


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_audio_capture_initialization_with_turn_taking(mock_vad, valid_audio_config, turn_taking_config):
    """Test AudioCapture initializes correctly with turn_taking_config."""
    # Mock the VAD constructor
    mock_vad_instance = MagicMock()
    mock_vad.Vad.return_value = mock_vad_instance
    
    capture = AudioCapture(
        valid_audio_config,
        turn_taking_config=turn_taking_config,
        default_duration=5.0
    )
    
    # Verify config values stored correctly
    assert capture.config == valid_audio_config
    assert capture.turn_taking_config == turn_taking_config
    assert capture.default_duration == 5.0
    
    # Verify VAD was created with correct aggressiveness level
    mock_vad.Vad.assert_called_once_with(turn_taking_config.vad_aggressiveness)
    assert capture.vad == mock_vad_instance


@patch('sovereign_core.ears.audio_capture.sd')
def test_capture_backward_compatibility(mock_sd, valid_audio_config):
    """Test old capture(duration) method still works without turn_taking_config."""
    duration = 3.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    mock_audio = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)
    
    mock_sd.rec.return_value = mock_audio
    mock_sd.wait.return_value = None
    
    # Initialize without turn_taking_config
    capture = AudioCapture(valid_audio_config, default_duration=duration)
    
    # Verify VAD is None
    assert capture.vad is None
    assert capture.turn_taking_config is None
    
    # Verify capture still works
    recording = capture.capture()
    
    assert isinstance(recording, AudioRecording)
    assert len(recording.audio_data) == num_samples
    assert recording.sample_rate == sample_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])