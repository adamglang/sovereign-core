"""Test Speech-to-Text functionality."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from sovereign_core.config import STTConfig
from sovereign_core.ears.audio_capture import AudioRecording
from sovereign_core.ears.speech_to_text import SpeechToText, TranscriptionResult


@pytest.fixture
def valid_stt_config():
    """Valid STT configuration."""
    return STTConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        model_dir="./models/whisper",
    )


@pytest.fixture
def cuda_stt_config():
    """STT configuration for CUDA."""
    return STTConfig(
        model_size="small",
        device="cuda",
        compute_type="float16",
        model_dir="./models/whisper",
    )


@pytest.fixture
def mock_audio_recording():
    """Mock audio recording for testing."""
    # Create 1 second of mock audio data (16kHz, mono)
    audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
    return AudioRecording(audio_data=audio_data, sample_rate=16000)


@pytest.fixture
def empty_audio_recording():
    """Empty audio recording for testing."""
    return AudioRecording(audio_data=np.array([], dtype=np.int16), sample_rate=16000)


def test_transcription_result_structure():
    """Test TranscriptionResult data structure."""
    result = TranscriptionResult(
        text="Hello world",
        confidence=0.95,
        language="en",
    )
    
    assert result.text == "Hello world"
    assert result.confidence == 0.95
    assert result.language == "en"


def test_stt_initialization(valid_stt_config):
    """Test SpeechToText initialization."""
    stt = SpeechToText(valid_stt_config)
    
    assert stt.config == valid_stt_config
    assert stt._model is None  # Lazy loading
    assert stt._device is None
    assert stt._compute_type is None


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_lazy_model_loading(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test model is loaded lazily on first transcription."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    # Mock transcription result
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    stt = SpeechToText(valid_stt_config)
    
    # Model should not be loaded yet
    assert stt._model is None
    mock_whisper_model.assert_not_called()
    
    # Trigger model loading with transcription
    stt.transcribe(mock_audio_recording)
    
    # Model should now be loaded
    assert stt._model is not None
    mock_whisper_model.assert_called_once()


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_successful_transcription(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test successful transcription with mocked WhisperModel."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    # Mock segments
    mock_segment1 = MagicMock()
    mock_segment1.text = "Hello"
    mock_segment1.avg_logprob = -0.2
    
    mock_segment2 = MagicMock()
    mock_segment2.text = "world"
    mock_segment2.avg_logprob = -0.15
    
    mock_info = MagicMock()
    mock_info.language = "en"
    
    mock_model_instance.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
    
    stt = SpeechToText(valid_stt_config)
    result = stt.transcribe(mock_audio_recording)
    
    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world"
    assert result.language == "en"
    assert 0.0 <= result.confidence <= 1.0


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_empty_audio_handling(mock_whisper_model, mock_torch, valid_stt_config, empty_audio_recording):
    """Test handling of empty audio data."""
    mock_torch.cuda.is_available.return_value = False
    
    stt = SpeechToText(valid_stt_config)
    result = stt.transcribe(empty_audio_recording)
    
    # Should return empty result without attempting transcription
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.language is None
    
    # Model should not be loaded or called
    mock_whisper_model.assert_not_called()


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_no_speech_detected(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test handling when no speech is detected in audio."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_info = MagicMock()
    mock_info.language = "en"
    
    # Return empty segments (no speech detected)
    mock_model_instance.transcribe.return_value = ([], mock_info)
    
    stt = SpeechToText(valid_stt_config)
    result = stt.transcribe(mock_audio_recording)
    
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.language == "en"


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_cpu_device_selection(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test CPU device is selected correctly."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    stt = SpeechToText(valid_stt_config)
    stt.transcribe(mock_audio_recording)
    
    # Verify model was loaded with CPU settings
    mock_whisper_model.assert_called_once_with(
        "base",
        device="cpu",
        compute_type="int8",
        download_root="./models/whisper",
    )


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_cuda_device_selection(mock_whisper_model, mock_torch, cuda_stt_config, mock_audio_recording):
    """Test CUDA device is selected when available."""
    mock_torch.cuda.is_available.return_value = True
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    stt = SpeechToText(cuda_stt_config)
    stt.transcribe(mock_audio_recording)
    
    # Verify model was loaded with CUDA settings
    mock_whisper_model.assert_called_once_with(
        "small",
        device="cuda",
        compute_type="float16",
        download_root="./models/whisper",
    )


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_cuda_fallback_to_cpu(mock_whisper_model, mock_torch, cuda_stt_config, mock_audio_recording):
    """Test fallback to CPU when CUDA is requested but unavailable."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    stt = SpeechToText(cuda_stt_config)
    stt.transcribe(mock_audio_recording)
    
    # Should fall back to CPU with int8
    mock_whisper_model.assert_called_once_with(
        "small",
        device="cpu",
        compute_type="int8",
        download_root="./models/whisper",
    )


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_audio_format_conversion(mock_whisper_model, mock_torch, valid_stt_config):
    """Test int16 audio is converted to float32 for transcription."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    # Create int16 audio data
    audio_int16 = np.array([16384, -16384, 32767, -32768], dtype=np.int16)
    recording = AudioRecording(audio_data=audio_int16, sample_rate=16000)
    
    stt = SpeechToText(valid_stt_config)
    stt.transcribe(recording)
    
    # Verify transcribe was called with float32 audio
    call_args = mock_model_instance.transcribe.call_args[0][0]
    assert call_args.dtype == np.float32
    
    # Verify normalization to [-1.0, 1.0]
    expected_float32 = audio_int16.astype(np.float32) / 32768.0
    np.testing.assert_array_almost_equal(call_args, expected_float32)


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_model_loading_failure(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test handling of model loading failures."""
    mock_torch.cuda.is_available.return_value = False
    mock_whisper_model.side_effect = RuntimeError("Model download failed")
    
    stt = SpeechToText(valid_stt_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        stt.transcribe(mock_audio_recording)
    
    assert "Whisper model loading failed" in str(exc_info.value)


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_transcription_failure(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test handling of transcription failures."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_model_instance.transcribe.side_effect = Exception("Transcription error")
    
    stt = SpeechToText(valid_stt_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        stt.transcribe(mock_audio_recording)
    
    assert "Failed to transcribe audio" in str(exc_info.value)


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_model_loaded_only_once(mock_whisper_model, mock_torch, valid_stt_config):
    """Test model is loaded only once for multiple transcriptions."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    # Create multiple recordings
    recording1 = AudioRecording(
        audio_data=np.random.randint(-100, 100, size=1000, dtype=np.int16),
        sample_rate=16000
    )
    recording2 = AudioRecording(
        audio_data=np.random.randint(-100, 100, size=1000, dtype=np.int16),
        sample_rate=16000
    )
    
    stt = SpeechToText(valid_stt_config)
    stt.transcribe(recording1)
    stt.transcribe(recording2)
    
    # Model should be loaded only once
    mock_whisper_model.assert_called_once()
    
    # But transcribe should be called twice
    assert mock_model_instance.transcribe.call_count == 2


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_confidence_calculation(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test confidence score calculation from log probabilities."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    # Create segments with known log probabilities
    mock_segment1 = MagicMock()
    mock_segment1.text = "test"
    mock_segment1.avg_logprob = -0.5  # exp(-0.5) ≈ 0.606
    
    mock_segment2 = MagicMock()
    mock_segment2.text = "text"
    mock_segment2.avg_logprob = -0.3  # exp(-0.3) ≈ 0.741
    
    mock_info = MagicMock()
    mock_info.language = "en"
    
    mock_model_instance.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
    
    stt = SpeechToText(valid_stt_config)
    result = stt.transcribe(mock_audio_recording)
    
    # Confidence should be the exponential of the average log probability
    expected_avg_logprob = (-0.5 + -0.3) / 2  # -0.4
    expected_confidence = np.exp(expected_avg_logprob)  # exp(-0.4) ≈ 0.670
    
    assert 0.0 <= result.confidence <= 1.0
    assert abs(result.confidence - expected_confidence) < 0.01


@patch('sovereign_core.ears.speech_to_text.torch')
@patch('sovereign_core.ears.speech_to_text.WhisperModel')
def test_context_manager(mock_whisper_model, mock_torch, valid_stt_config, mock_audio_recording):
    """Test SpeechToText works as context manager."""
    mock_torch.cuda.is_available.return_value = False
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    
    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.1
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    with SpeechToText(valid_stt_config) as stt:
        assert isinstance(stt, SpeechToText)
        result = stt.transcribe(mock_audio_recording)
        assert isinstance(result, TranscriptionResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])