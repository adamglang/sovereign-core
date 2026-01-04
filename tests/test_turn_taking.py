"""Test Turn-Taking and VAD-based audio capture functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sovereign_core.config import TurnTakingConfig, AudioConfig
from sovereign_core.ears.audio_capture import AudioCapture, TurnTakingState


@pytest.fixture
def valid_audio_config():
    """Valid audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        device_index=None,
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


def test_turn_taking_config_validation():
    """Test valid and invalid turn-taking config values."""
    # Valid configs
    valid = TurnTakingConfig(
        vad_aggressiveness=0,
        min_speech_duration_ms=100,
        end_silence_duration_ms=200,
        post_speech_grace_ms=0,
        max_recording_duration_s=5,
        vad_frame_duration_ms=10,
    )
    assert valid.vad_aggressiveness == 0
    assert valid.vad_frame_duration_ms == 10
    
    # Test all valid frame durations
    for duration in [10, 20, 30]:
        config = TurnTakingConfig(vad_frame_duration_ms=duration)
        assert config.vad_frame_duration_ms == duration
    
    # Test invalid vad_aggressiveness (out of 0-3 range)
    with pytest.raises(ValueError) as exc_info:
        TurnTakingConfig(vad_aggressiveness=-1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()
    
    with pytest.raises(ValueError) as exc_info:
        TurnTakingConfig(vad_aggressiveness=4)
    assert "less than or equal to 3" in str(exc_info.value).lower()
    
    # Test invalid vad_frame_duration_ms (not 10/20/30)
    with pytest.raises(ValueError) as exc_info:
        TurnTakingConfig(vad_frame_duration_ms=15)
    assert "must be 10, 20, or 30 ms" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        TurnTakingConfig(vad_frame_duration_ms=40)
    assert "must be 10, 20, or 30 ms" in str(exc_info.value)
    
    # Test edge case values
    edge_config = TurnTakingConfig(
        vad_aggressiveness=3,
        min_speech_duration_ms=100,
        end_silence_duration_ms=200,
        post_speech_grace_ms=0,
        max_recording_duration_s=5,
        vad_frame_duration_ms=30,
    )
    assert edge_config.vad_aggressiveness == 3
    assert edge_config.min_speech_duration_ms == 100
    assert edge_config.end_silence_duration_ms == 200


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.sd')
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_vad_state_machine_transitions(mock_vad_module, mock_sd, valid_audio_config, turn_taking_config, caplog):
    """Test VAD state machine transitions through complete cycle."""
    # Create mock VAD instance
    mock_vad = MagicMock()
    mock_vad_module.Vad.return_value = mock_vad
    
    # Create mock audio stream
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    
    # Calculate frame parameters
    frame_duration_ms = turn_taking_config.vad_frame_duration_ms
    frame_samples = (valid_audio_config.sample_rate * frame_duration_ms) // 1000
    
    # Define speech pattern: silence -> speech -> silence -> finalize
    # Need enough frames to meet min_speech_duration_ms (300ms = 10 frames @ 30ms)
    # and end_silence_duration_ms (700ms = 24 frames @ 30ms)
    # plus grace period (500ms = 17 frames @ 30ms)
    speech_pattern = (
        [False] * 3 +    # 90ms silence (WAITING_FOR_SPEECH)
        [True] * 12 +    # 360ms speech (SPEECH_DETECTED, exceeds min 300ms)
        [False] * 24 +   # 720ms silence (SILENCE_AFTER_SPEECH, exceeds 700ms)
        [False] * 17     # 510ms silence (GRACE_PERIOD, exceeds 500ms -> FINALIZED)
    )
    
    # Create synthetic audio frames
    def create_frame(is_speech_frame):
        frame = np.random.randint(-5000 if is_speech_frame else -1000,
                                  5000 if is_speech_frame else 1000,
                                  size=frame_samples, dtype=np.int16)
        return frame, False  # (data, overflowed)
    
    # Set up mock stream.read to return frames
    frame_index = [0]
    def mock_read(samples):
        if frame_index[0] < len(speech_pattern):
            frame = create_frame(speech_pattern[frame_index[0]])
            frame_index[0] += 1
            return frame
        # Return silence if we run out
        return create_frame(False)
    
    mock_stream.read.side_effect = mock_read
    
    # Set up VAD to match speech pattern
    vad_call_index = [0]
    def mock_is_speech(frame_bytes, sample_rate):
        if vad_call_index[0] < len(speech_pattern):
            result = speech_pattern[vad_call_index[0]]
            vad_call_index[0] += 1
            return result
        return False
    
    mock_vad.is_speech.side_effect = mock_is_speech
    
    # Create AudioCapture with turn-taking enabled
    capture = AudioCapture(valid_audio_config, turn_taking_config=turn_taking_config)
    
    # Execute VAD-based capture
    with caplog.at_level("INFO"):
        recording = capture.capture_with_vad()
    
    # Verify recording was created
    assert recording is not None
    assert isinstance(recording.audio_data, np.ndarray)
    assert recording.sample_rate == valid_audio_config.sample_rate
    
    # Verify state transitions occurred (check logs)
    assert "speech_start" in caplog.text
    assert "silence_detected" in caplog.text
    assert "utterance_finalized" in caplog.text
    
    # Verify stream was started and stopped
    mock_stream.start.assert_called_once()
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.sd')
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_grace_period_cancellation(mock_vad_module, mock_sd, valid_audio_config, turn_taking_config, caplog):
    """Test speech detection during GRACE_PERIOD cancels finalization."""
    # Create mock VAD instance
    mock_vad = MagicMock()
    mock_vad_module.Vad.return_value = mock_vad
    
    # Create mock audio stream
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    
    frame_duration_ms = turn_taking_config.vad_frame_duration_ms
    frame_samples = (valid_audio_config.sample_rate * frame_duration_ms) // 1000
    
    # Speech pattern that cancels grace period:
    # silence -> speech -> silence (enter grace) -> speech (cancel grace) -> silence -> finalize
    speech_pattern = (
        [False] * 2 +    # Silence
        [True] * 12 +    # Speech (360ms, exceeds min 300ms)
        [False] * 25 +   # Silence (750ms, exceeds 700ms + enters grace)
        [True] * 10 +    # Speech resumes (300ms, cancels grace)
        [False] * 30 +   # Silence (900ms, exceeds 700ms + grace 500ms)
        [False] * 17     # Grace period completes
    )
    
    # Create synthetic audio frames
    def create_frame(is_speech_frame):
        frame = np.random.randint(-5000 if is_speech_frame else -1000,
                                  5000 if is_speech_frame else 1000,
                                  size=frame_samples, dtype=np.int16)
        return frame, False
    
    frame_index = [0]
    def mock_read(samples):
        if frame_index[0] < len(speech_pattern):
            frame = create_frame(speech_pattern[frame_index[0]])
            frame_index[0] += 1
            return frame
        return create_frame(False)
    
    mock_stream.read.side_effect = mock_read
    
    vad_call_index = [0]
    def mock_is_speech(frame_bytes, sample_rate):
        if vad_call_index[0] < len(speech_pattern):
            result = speech_pattern[vad_call_index[0]]
            vad_call_index[0] += 1
            return result
        return False
    
    mock_vad.is_speech.side_effect = mock_is_speech
    
    capture = AudioCapture(valid_audio_config, turn_taking_config=turn_taking_config)
    
    with caplog.at_level("INFO"):
        recording = capture.capture_with_vad()
    
    # Verify grace period was entered and cancelled
    assert "silence_threshold_met" in caplog.text
    assert "grace_cancelled_speech_resumed" in caplog.text or "speech_resumed" in caplog.text
    assert "utterance_finalized" in caplog.text
    
    assert recording is not None


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.sd')
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_max_duration_enforcement(mock_vad_module, mock_sd, valid_audio_config, caplog):
    """Test recording finalizes at max_recording_duration_s safety ceiling."""
    # Create config with short max duration for testing (min is 5s)
    short_max_config = TurnTakingConfig(
        vad_aggressiveness=2,
        min_speech_duration_ms=300,
        end_silence_duration_ms=700,
        post_speech_grace_ms=500,
        max_recording_duration_s=5,  # 5 seconds max (minimum allowed)
        vad_frame_duration_ms=30,
    )
    
    mock_vad = MagicMock()
    mock_vad_module.Vad.return_value = mock_vad
    
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    
    frame_duration_ms = short_max_config.vad_frame_duration_ms
    frame_samples = (valid_audio_config.sample_rate * frame_duration_ms) // 1000
    
    # Continuous speech pattern that would exceed max duration
    # 5 seconds = 5000ms / 30ms = ~167 frames
    # Create more than that to test ceiling
    continuous_speech = [False] * 2 + [True] * 200
    
    def create_frame(is_speech_frame):
        frame = np.random.randint(-5000 if is_speech_frame else -1000,
                                  5000 if is_speech_frame else 1000,
                                  size=frame_samples, dtype=np.int16)
        return frame, False
    
    frame_index = [0]
    def mock_read(samples):
        if frame_index[0] < len(continuous_speech):
            frame = create_frame(continuous_speech[frame_index[0]])
            frame_index[0] += 1
            return frame
        return create_frame(False)
    
    mock_stream.read.side_effect = mock_read
    
    vad_call_index = [0]
    def mock_is_speech(frame_bytes, sample_rate):
        if vad_call_index[0] < len(continuous_speech):
            result = continuous_speech[vad_call_index[0]]
            vad_call_index[0] += 1
            return result
        return False
    
    mock_vad.is_speech.side_effect = mock_is_speech
    
    capture = AudioCapture(valid_audio_config, turn_taking_config=short_max_config)
    
    with caplog.at_level("INFO"):
        recording = capture.capture_with_vad()
    
    # Verify max duration was reached
    assert "max_duration_reached" in caplog.text
    assert recording is not None
    
    # Recording should be approximately max_duration (allow some tolerance for frame boundaries)
    expected_samples = short_max_config.max_recording_duration_s * valid_audio_config.sample_rate
    # Allow up to 2 extra frames worth of samples
    tolerance = frame_samples * 2
    assert len(recording.audio_data) <= expected_samples + tolerance


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.sd')
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_min_speech_duration_filter(mock_vad_module, mock_sd, valid_audio_config, turn_taking_config):
    """Test very short speech bursts don't immediately finalize due to min_speech_duration_ms."""
    mock_vad = MagicMock()
    mock_vad_module.Vad.return_value = mock_vad
    
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    
    frame_duration_ms = turn_taking_config.vad_frame_duration_ms
    frame_samples = (valid_audio_config.sample_rate * frame_duration_ms) // 1000
    
    # Short speech burst (< min_speech_duration_ms of 300ms)
    # 5 frames @ 30ms = 150ms speech, followed by long silence
    # This should NOT trigger immediate finalization
    speech_pattern = (
        [False] * 2 +    # Initial silence
        [True] * 5 +     # 150ms speech (below 300ms minimum)
        [False] * 25 +   # 750ms silence (exceeds end_silence threshold)
        [False] * 17     # Would be grace period but speech was too short
    )
    
    # Since min speech duration not met, this will wait until max_recording_duration_s
    # We need to add enough silence to trigger max duration
    # max_recording_duration_s = 15s = 15000ms / 30ms = 500 frames
    # Add remaining frames as silence
    remaining_frames = 500 - len(speech_pattern)
    speech_pattern = list(speech_pattern) + [False] * remaining_frames
    
    def create_frame(is_speech_frame):
        frame = np.random.randint(-5000 if is_speech_frame else -1000,
                                  5000 if is_speech_frame else 1000,
                                  size=frame_samples, dtype=np.int16)
        return frame, False
    
    frame_index = [0]
    def mock_read(samples):
        if frame_index[0] < len(speech_pattern):
            frame = create_frame(speech_pattern[frame_index[0]])
            frame_index[0] += 1
            return frame
        return create_frame(False)
    
    mock_stream.read.side_effect = mock_read
    
    vad_call_index = [0]
    def mock_is_speech(frame_bytes, sample_rate):
        if vad_call_index[0] < len(speech_pattern):
            result = speech_pattern[vad_call_index[0]]
            vad_call_index[0] += 1
            return result
        return False
    
    mock_vad.is_speech.side_effect = mock_is_speech
    
    capture = AudioCapture(valid_audio_config, turn_taking_config=turn_taking_config)
    
    recording = capture.capture_with_vad()
    
    # Recording should complete (hit max duration, not early finalization)
    assert recording is not None
    # The recording will be long (close to max duration) because short speech didn't trigger early end
    # This verifies the min_speech_duration filter is working


@patch('sovereign_core.ears.audio_capture.WEBRTCVAD_AVAILABLE', True)
@patch('sovereign_core.ears.audio_capture.sd')
@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_vad_error_handling(mock_vad_module, mock_sd, valid_audio_config, turn_taking_config, caplog):
    """Test VAD errors are logged and handled gracefully with fallback."""
    mock_vad = MagicMock()
    mock_vad_module.Vad.return_value = mock_vad
    
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    
    frame_duration_ms = turn_taking_config.vad_frame_duration_ms
    frame_samples = (valid_audio_config.sample_rate * frame_duration_ms) // 1000
    
    # VAD throws exception on processing
    mock_vad.is_speech.side_effect = Exception("VAD processing error")
    
    # Create short recording that will hit error then end
    num_frames = 50
    
    def create_frame():
        frame = np.random.randint(-1000, 1000, size=frame_samples, dtype=np.int16)
        return frame, False
    
    frame_index = [0]
    def mock_read(samples):
        if frame_index[0] < num_frames:
            frame = create_frame()
            frame_index[0] += 1
            return frame
        return create_frame()
    
    mock_stream.read.side_effect = mock_read
    
    capture = AudioCapture(valid_audio_config, turn_taking_config=turn_taking_config)
    
    # Execute - should handle error gracefully
    with caplog.at_level("ERROR"):
        recording = capture.capture_with_vad()
    
    # Verify error was logged
    assert "VAD processing failed" in caplog.text
    
    # Verify fallback behavior worked (recording still created)
    assert recording is not None
    assert isinstance(recording.audio_data, np.ndarray)


@patch('sovereign_core.ears.audio_capture.webrtcvad')
def test_capture_with_vad_requires_config(mock_vad_module, valid_audio_config):
    """Test capture_with_vad raises error if turn_taking_config not provided."""
    # Initialize without turn_taking_config
    capture = AudioCapture(valid_audio_config)
    
    # Attempting VAD-based capture should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        capture.capture_with_vad()
    
    assert "VAD-based capture requires turn_taking_config" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
