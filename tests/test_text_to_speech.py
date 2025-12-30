"""Test Text-to-Speech functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sovereign_core.config import TTSConfig
from sovereign_core.mouth.text_to_speech import TextToSpeech


@pytest.fixture
def valid_tts_config():
    """Valid TTS configuration."""
    return TTSConfig(
        provider="windows",
        voice=None,
        rate=1.0,
    )


@pytest.fixture
def custom_tts_config():
    """TTS configuration with custom settings."""
    return TTSConfig(
        provider="windows",
        voice="David",
        rate=1.5,
    )


@pytest.fixture
def mock_pyttsx3_engine():
    """Mock pyttsx3 engine."""
    mock_engine = MagicMock()
    mock_engine.getProperty.side_effect = lambda prop: {
        'rate': 200,
        'volume': 1.0,
        'voices': [],
    }.get(prop)
    return mock_engine


def test_tts_initialization(valid_tts_config):
    """Test TextToSpeech initialization."""
    tts = TextToSpeech(valid_tts_config)
    
    assert tts.config == valid_tts_config
    assert tts._engine is None  # Lazy loading


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_lazy_engine_loading(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test TTS engine is loaded lazily on first speak."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    
    # Engine should not be loaded yet
    assert tts._engine is None
    mock_pyttsx3.init.assert_not_called()
    
    # Trigger engine loading with speak
    tts.speak("Hello")
    
    # Engine should now be loaded
    assert tts._engine is not None
    mock_pyttsx3.init.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_successful_speak(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test successful text-to-speech with mocked pyttsx3."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Hello world")
    
    # Verify engine methods were called
    mock_pyttsx3_engine.say.assert_called_once_with("Hello world")
    mock_pyttsx3_engine.runAndWait.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_empty_text_handling(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test handling of empty text input."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    
    # Empty string should not trigger speech
    tts.speak("")
    mock_pyttsx3_engine.say.assert_not_called()
    
    # Whitespace-only should not trigger speech
    tts.speak("   ")
    mock_pyttsx3_engine.say.assert_not_called()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_voice_selection(mock_pyttsx3, custom_tts_config):
    """Test custom voice selection."""
    mock_engine = MagicMock()
    mock_pyttsx3.init.return_value = mock_engine
    
    # Create mock voices
    mock_voice1 = MagicMock()
    mock_voice1.name = "Microsoft David Desktop"
    mock_voice1.id = "voice_id_1"
    
    mock_voice2 = MagicMock()
    mock_voice2.name = "Microsoft Zira Desktop"
    mock_voice2.id = "voice_id_2"
    
    mock_engine.getProperty.side_effect = lambda prop: {
        'rate': 200,
        'voices': [mock_voice1, mock_voice2],
    }.get(prop)
    
    tts = TextToSpeech(custom_tts_config)
    tts.speak("Test")
    
    # Verify voice was set (David matches mock_voice1)
    mock_engine.setProperty.assert_any_call('voice', 'voice_id_1')


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_voice_not_found_fallback(mock_pyttsx3, mock_pyttsx3_engine):
    """Test fallback to default voice when specified voice not found."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    # Create mock voices without the requested voice
    mock_voice = MagicMock()
    mock_voice.name = "Microsoft Zira Desktop"
    mock_voice.id = "voice_id_1"
    
    def get_property(prop):
        if prop == 'voices':
            return [mock_voice]
        elif prop == 'rate':
            return 200
        return None
    
    mock_pyttsx3_engine.getProperty.side_effect = get_property
    
    config = TTSConfig(provider="windows", voice="NonexistentVoice", rate=1.0)
    tts = TextToSpeech(config)
    tts.speak("Test")
    
    # Should not have set voice property (falls back to default)
    voice_set_calls = [
        call for call in mock_pyttsx3_engine.setProperty.call_args_list
        if call[0][0] == 'voice'
    ]
    assert len(voice_set_calls) == 0


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_speech_rate_setting(mock_pyttsx3, custom_tts_config, mock_pyttsx3_engine):
    """Test speech rate is set correctly."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(custom_tts_config)
    tts.speak("Test")
    
    # Rate should be default (200) * config rate (1.5) = 300
    mock_pyttsx3_engine.setProperty.assert_any_call('rate', 300)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_volume_setting(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test volume is set to maximum."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("Test")
    
    # Volume should be set to 1.0 (max)
    mock_pyttsx3_engine.setProperty.assert_any_call('volume', 1.0)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_engine_initialization_failure(mock_pyttsx3, valid_tts_config):
    """Test handling of TTS engine initialization failures."""
    mock_pyttsx3.init.side_effect = RuntimeError("Engine init failed")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        tts.speak("Test")
    
    assert "TTS engine initialization failed" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_speech_synthesis_failure(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test handling of speech synthesis failures."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.runAndWait.side_effect = RuntimeError("Speech failed")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        tts.speak("Test")
    
    assert "Failed to speak text" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_engine_loaded_only_once(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test engine is loaded only once for multiple speak calls."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("First")
    tts.speak("Second")
    tts.speak("Third")
    
    # Engine should be initialized only once
    mock_pyttsx3.init.assert_called_once()
    
    # But speak should be called three times
    assert mock_pyttsx3_engine.say.call_count == 3
    assert mock_pyttsx3_engine.runAndWait.call_count == 3


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
@pytest.mark.asyncio
async def test_speak_async(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test asynchronous speak functionality."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    await tts.speak_async("Hello async")
    
    # Verify engine methods were called
    mock_pyttsx3_engine.say.assert_called_once_with("Hello async")
    mock_pyttsx3_engine.runAndWait.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
@pytest.mark.asyncio
async def test_speak_async_empty_text(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test async speak handles empty text."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    await tts.speak_async("")
    
    # Should not trigger speech
    mock_pyttsx3_engine.say.assert_not_called()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
@pytest.mark.asyncio
async def test_speak_async_failure(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test async speak handles failures."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.runAndWait.side_effect = RuntimeError("Async speech failed")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        await tts.speak_async("Test")
    
    assert "Failed to speak text asynchronously" in str(exc_info.value)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_context_manager(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test TextToSpeech works as context manager."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    with TextToSpeech(valid_tts_config) as tts:
        assert isinstance(tts, TextToSpeech)
        tts.speak("Test")
    
    # Verify cleanup was called
    mock_pyttsx3_engine.stop.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_context_manager_cleanup_error(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test context manager handles cleanup errors gracefully."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.stop.side_effect = Exception("Stop error")
    
    # Should not raise even if stop fails
    with TextToSpeech(valid_tts_config) as tts:
        tts.speak("Test")


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_context_manager_without_speak(mock_pyttsx3, valid_tts_config):
    """Test context manager when engine was never initialized."""
    with TextToSpeech(valid_tts_config) as tts:
        # Don't call speak, so engine never initializes
        pass
    
    # Should not raise even though engine is None


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_multiple_text_inputs(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test speaking multiple different texts."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    
    texts = [
        "First message",
        "Second message with more words",
        "Third!",
    ]
    
    for text in texts:
        tts.speak(text)
    
    # Verify all texts were spoken
    assert mock_pyttsx3_engine.say.call_count == len(texts)
    for i, text in enumerate(texts):
        assert mock_pyttsx3_engine.say.call_args_list[i][0][0] == text


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_long_text_handling(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test handling of long text inputs."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    long_text = "This is a very long sentence. " * 100
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak(long_text)
    
    # Should handle long text without issues
    mock_pyttsx3_engine.say.assert_called_once_with(long_text)
    mock_pyttsx3_engine.runAndWait.assert_called_once()


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_special_characters_in_text(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """Test handling of special characters in text."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    special_text = "Hello! How are you? I'm fine. #test @user $100 50%"
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak(special_text)
    
    # Should handle special characters
    mock_pyttsx3_engine.say.assert_called_once_with(special_text)


@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_default_config_values(mock_pyttsx3, mock_pyttsx3_engine):
    """Test TTS with default configuration values."""
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    config = TTSConfig()  # Use all defaults
    tts = TextToSpeech(config)
    tts.speak("Test")
    
    # Verify defaults were applied
    assert config.provider == "windows"
    assert config.voice is None
    assert config.rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])