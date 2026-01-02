"""Test Text-to-Speech functionality."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

# Mock piper modules before import
piper_mock = MagicMock()
piper_config_mock = MagicMock()
sys.modules['piper'] = piper_mock
sys.modules['piper.config'] = piper_config_mock

from sovereign_core.config import TTSConfig
from sovereign_core.mouth.text_to_speech import TextToSpeech


@pytest.fixture
def valid_tts_config():
    """Valid TTS configuration with Piper settings."""
    return TTSConfig(
        voice_model="en_US-amy-medium",
        speaker_id=0,
        use_cuda=False,
    )


@pytest.fixture
def cuda_tts_config():
    """TTS configuration with CUDA enabled."""
    return TTSConfig(
        voice_model="en_US-amy-medium",
        speaker_id=0,
        use_cuda=True,
    )


@pytest.fixture
def mock_piper_voice():
    """Mock PiperVoice instance with synthesize method."""
    mock_voice = Mock()
    
    def create_audio_chunk():
        """Create a fresh mock AudioChunk each time."""
        mock_chunk = Mock()
        mock_chunk.audio_int16_array = np.zeros(22050, dtype=np.int16)
        mock_chunk.sample_rate = 22050
        return mock_chunk
    
    # Return a fresh iterator each time synthesize is called
    mock_voice.synthesize.side_effect = lambda *args, **kwargs: iter([create_audio_chunk()])
    mock_voice.config.sample_rate = 22050
    return mock_voice


@pytest.fixture
def mock_piper_class(mock_piper_voice):
    """Mock PiperVoice class."""
    with patch('sovereign_core.mouth.text_to_speech.PiperVoice') as mock_class:
        mock_class.load.return_value = mock_piper_voice
        yield mock_class


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    with patch('sovereign_core.mouth.text_to_speech.sd') as mock_sd:
        yield mock_sd


@pytest.fixture
def mock_path_exists():
    """Mock Path.exists() to return True for model files."""
    with patch.object(Path, 'exists', return_value=True):
        yield


def test_tts_initialization(valid_tts_config):
    """Test TextToSpeech initialization."""
    tts = TextToSpeech(valid_tts_config)
    
    assert tts.config == valid_tts_config
    assert tts.voice is None


def test_lazy_voice_loading(valid_tts_config, mock_piper_class, mock_sounddevice, mock_path_exists):
    """Test voice is loaded lazily on first speak."""
    tts = TextToSpeech(valid_tts_config)
    
    assert tts.voice is None
    mock_piper_class.load.assert_not_called()
    
    tts.speak("Hello world")
    
    assert tts.voice is not None
    mock_piper_class.load.assert_called_once()


def test_voice_loaded_only_once(valid_tts_config, mock_piper_class, mock_sounddevice, mock_path_exists):
    """Test voice is loaded only once across multiple speaks."""
    tts = TextToSpeech(valid_tts_config)
    
    tts.speak("First")
    tts.speak("Second")
    tts.speak("Third")
    
    mock_piper_class.load.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_speak_basic(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test successful text-to-speech with mocked Piper."""
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Hello world")
    
    # Verify synthesize called with text and SynthesisConfig
    assert mock_piper_voice.synthesize.call_count == 1
    call_args = mock_piper_voice.synthesize.call_args[0]
    assert call_args[0] == "Hello world"
    # Check that SynthesisConfig was passed (not None since speaker_id=0)
    assert call_args[1] is not None
    
    mock_play.assert_called_once()
    mock_wait.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_speak_with_speaker_id(mock_wait, mock_play, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test speaking with custom speaker ID."""
    config = TTSConfig(voice_model="en_US-amy-medium", speaker_id=3, use_cuda=False)
    tts = TextToSpeech(config)
    tts.speak("Test")
    
    # Verify SynthesisConfig created with speaker_id
    assert mock_piper_voice.synthesize.call_count == 1
    call_args = mock_piper_voice.synthesize.call_args[0]
    assert call_args[0] == "Test"
    assert call_args[1] is not None  # SynthesisConfig should be passed


def test_empty_text_handling(valid_tts_config, mock_piper_class, mock_sounddevice, mock_path_exists):
    """Test handling of empty text input."""
    tts = TextToSpeech(valid_tts_config)
    
    tts.speak("")
    mock_piper_class.load.assert_not_called()
    
    tts.speak("   ")
    mock_piper_class.load.assert_not_called()


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_cuda_loading(mock_wait, mock_play, cuda_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test voice loading with CUDA enabled."""
    tts = TextToSpeech(cuda_tts_config)
    tts.speak("Test")
    
    call_args = mock_piper_class.load.call_args
    assert call_args[1]['use_cuda'] is True
    assert 'en_US-amy-medium.onnx' in call_args[0][0]


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_cpu_loading(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test voice loading with CPU (CUDA disabled)."""
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Test")
    
    call_args = mock_piper_class.load.call_args
    assert call_args[1]['use_cuda'] is False
    assert 'en_US-amy-medium.onnx' in call_args[0][0]


def test_missing_model_files(valid_tts_config, mock_piper_class):
    """Test FileNotFoundError when model files are missing."""
    with patch.object(Path, 'exists', return_value=False):
        tts = TextToSpeech(valid_tts_config)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            tts.speak("Test")
        
        assert "Piper model files not found" in str(exc_info.value)
        assert "SETUP.md" in str(exc_info.value)


def test_voice_loading_failure(valid_tts_config, mock_piper_class, mock_path_exists):
    """Test handling of voice loading failures."""
    mock_piper_class.load.side_effect = RuntimeError("CUDA unavailable")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        tts.speak("Test")
    
    assert "Voice model loading failed" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_synthesis_failure(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test handling of speech synthesis failures."""
    mock_piper_voice.synthesize.side_effect = RuntimeError("Synthesis error")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        tts.speak("Test")
    
    assert "Failed to speak text" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_playback_failure(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test handling of audio playback failures."""
    mock_play.side_effect = RuntimeError("Audio device error")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        tts.speak("Test")
    
    assert "Failed to speak text" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_speak_async(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test asynchronous speech delegates to sync speak."""
    import asyncio
    
    tts = TextToSpeech(valid_tts_config)
    
    asyncio.run(tts.speak_async("Async test"))
    
    # Verify synthesize was called
    assert mock_piper_voice.synthesize.call_count == 1
    call_args = mock_piper_voice.synthesize.call_args[0]
    assert call_args[0] == "Async test"
    
    mock_play.assert_called_once()
    mock_wait.assert_called_once()


def test_speak_async_empty_text(valid_tts_config, mock_piper_class, mock_sounddevice, mock_path_exists):
    """Test async speak with empty text."""
    import asyncio
    
    tts = TextToSpeech(valid_tts_config)
    
    asyncio.run(tts.speak_async(""))
    mock_piper_class.load.assert_not_called()
    
    asyncio.run(tts.speak_async("   "))
    mock_piper_class.load.assert_not_called()


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_speak_async_failure(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test handling of async speech failures."""
    import asyncio
    
    mock_piper_voice.synthesize.side_effect = RuntimeError("Async synthesis error")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(tts.speak_async("Test"))
    
    assert "Failed to speak text asynchronously" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_cleanup(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test cleanup releases resources."""
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Test")
    
    assert tts.voice is not None
    
    tts.cleanup()
    
    assert tts.voice is None


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_context_manager(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test TextToSpeech works as context manager."""
    with TextToSpeech(valid_tts_config) as tts:
        assert isinstance(tts, TextToSpeech)
        tts.speak("Test")
        assert tts.voice is not None
    
    assert tts.voice is None


def test_context_manager_without_speak(valid_tts_config):
    """Test context manager when voice was never loaded."""
    with TextToSpeech(valid_tts_config) as tts:
        pass
    
    assert tts.voice is None


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_multiple_text_inputs(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test speaking multiple different texts."""
    tts = TextToSpeech(valid_tts_config)
    
    texts = [
        "First message",
        "Second message with more words",
        "Third!",
    ]
    
    for text in texts:
        tts.speak(text)
    
    assert mock_piper_voice.synthesize.call_count == len(texts)
    # Check that all texts were synthesized
    for i, text in enumerate(texts):
        assert mock_piper_voice.synthesize.call_args_list[i][0][0] == text


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_long_text_handling(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test handling of long text inputs."""
    long_text = "This is a very long sentence. " * 100
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak(long_text)
    
    # Verify long text was synthesized
    assert mock_piper_voice.synthesize.call_count == 1
    assert mock_piper_voice.synthesize.call_args[0][0] == long_text
    mock_play.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_special_characters_in_text(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test handling of special characters in text."""
    special_text = "Hello! How are you? I'm fine. #test @user $100 50%"
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak(special_text)
    
    # Verify special text was synthesized
    assert mock_piper_voice.synthesize.call_count == 1
    assert mock_piper_voice.synthesize.call_args[0][0] == special_text


def test_default_config_values():
    """Test TTS with default configuration values."""
    config = TTSConfig()
    tts = TextToSpeech(config)
    
    assert config.voice_model == "en_US-lessac-medium"
    assert config.speaker_id is None
    assert config.use_cuda is True


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_audio_array_conversion(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test audio data is properly converted to numpy array."""
    # Create mock AudioChunks
    mock_chunk1 = Mock()
    mock_chunk1.audio_int16_array = np.array([1, 2, 3], dtype=np.int16)
    mock_chunk1.sample_rate = 22050
    
    mock_chunk2 = Mock()
    mock_chunk2.audio_int16_array = np.array([4, 5, 6], dtype=np.int16)
    mock_chunk2.sample_rate = 22050
    
    # Override fixture's side_effect with custom chunks
    mock_piper_voice.synthesize.side_effect = lambda *args, **kwargs: iter([mock_chunk1, mock_chunk2])
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Test")
    
    called_audio = mock_play.call_args[0][0]
    
    assert isinstance(called_audio, np.ndarray)
    assert called_audio.dtype == np.int16
    # Should concatenate into single array [1,2,3,4,5,6]
    assert len(called_audio) == 6
    np.testing.assert_array_equal(called_audio, np.array([1, 2, 3, 4, 5, 6], dtype=np.int16))


@patch('sovereign_core.mouth.text_to_speech.sd.play')
@patch('sovereign_core.mouth.text_to_speech.sd.wait')
def test_sample_rate_from_voice_config(mock_wait, mock_play, valid_tts_config, mock_piper_class, mock_piper_voice, mock_path_exists):
    """Test sample rate is taken from AudioChunk, not voice config."""
    # Create chunk with custom sample rate
    def create_custom_chunk():
        mock_chunk = Mock()
        mock_chunk.audio_int16_array = np.zeros(22050, dtype=np.int16)
        mock_chunk.sample_rate = 44100  # Custom sample rate
        return mock_chunk
    
    mock_piper_voice.synthesize.side_effect = lambda *args, **kwargs: iter([create_custom_chunk()])
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Test")
    
    mock_play.assert_called_once()
    assert mock_play.call_args[1]['samplerate'] == 44100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
