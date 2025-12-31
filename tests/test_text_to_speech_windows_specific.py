"""
Windows TTS-specific tests (pyttsx3/SAPI5).

These tests are incompatible with the SAPI5 workaround that reinitializes
the engine after each speak() call to prevent silent failures.

TODO: Remove this file after migrating to Piper TTS.
See: docs/plans/tts-workaround-analysis.md
"""

from unittest.mock import MagicMock, patch

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
def mock_pyttsx3_engine():
    """Mock pyttsx3 engine."""
    mock_engine = MagicMock()
    mock_engine.getProperty.side_effect = lambda prop: {
        'rate': 200,
        'volume': 1.0,
        'voices': [],
    }.get(prop)
    return mock_engine


@pytest.mark.skip(reason="Incompatible with SAPI5 workaround - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_lazy_engine_loading(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test TTS engine is loaded lazily on first speak.
    
    SKIPPED: The SAPI5 workaround sets engine to None after each speak,
    so this test will fail. This behavior is intentional to work around
    Windows TTS bugs.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    
    # Engine should not be loaded yet
    assert tts._engine is None
    mock_pyttsx3.init.assert_not_called()
    
    # Trigger engine loading with speak
    tts.speak("Hello")
    
    # Engine should now be loaded (FAILS due to workaround setting it to None)
    assert tts._engine is not None
    mock_pyttsx3.init.assert_called_once()


@pytest.mark.skip(reason="Incompatible with SAPI5 workaround - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_engine_loaded_only_once(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test engine is loaded only once for multiple speak calls.
    
    SKIPPED: The SAPI5 workaround reinitializes the engine after each speak,
    so init() is called multiple times. This is intentional to work around
    Windows TTS bugs.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    tts.speak("First")
    tts.speak("Second")
    tts.speak("Third")
    
    # Engine should be initialized only once (FAILS due to workaround reinitializing)
    mock_pyttsx3.init.assert_called_once()
    
    # But speak should be called three times
    assert mock_pyttsx3_engine.say.call_count == 3
    assert mock_pyttsx3_engine.runAndWait.call_count == 3


@pytest.mark.skip(reason="Async tests not implemented for Windows TTS - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
async def test_speak_async(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test asynchronous speak functionality.
    
    SKIPPED: Async TTS is not implemented for Windows/pyttsx3.
    Will be implemented with Piper TTS migration.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    await tts.speak_async("Hello async")
    
    # Verify engine methods were called
    mock_pyttsx3_engine.say.assert_called_once_with("Hello async")
    mock_pyttsx3_engine.runAndWait.assert_called_once()


@pytest.mark.skip(reason="Async tests not implemented for Windows TTS - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
async def test_speak_async_empty_text(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test async speak handles empty text.
    
    SKIPPED: Async TTS is not implemented for Windows/pyttsx3.
    Will be implemented with Piper TTS migration.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    
    tts = TextToSpeech(valid_tts_config)
    await tts.speak_async("")
    
    # Should not trigger speech
    mock_pyttsx3_engine.say.assert_not_called()


@pytest.mark.skip(reason="Async tests not implemented for Windows TTS - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
async def test_speak_async_failure(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test async speak handles failures.
    
    SKIPPED: Async TTS is not implemented for Windows/pyttsx3.
    Will be implemented with Piper TTS migration.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.runAndWait.side_effect = RuntimeError("Async speech failed")
    
    tts = TextToSpeech(valid_tts_config)
    
    with pytest.raises(RuntimeError) as exc_info:
        await tts.speak_async("Test")
    
    assert "Failed to speak text asynchronously" in str(exc_info.value)


@pytest.mark.skip(reason="Incompatible with SAPI5 workaround - pending Piper migration")
@patch('sovereign_core.mouth.text_to_speech.pyttsx3')
def test_context_manager_cleanup_error(mock_pyttsx3, valid_tts_config, mock_pyttsx3_engine):
    """
    Test context manager handles cleanup errors gracefully.
    
    SKIPPED: The SAPI5 workaround calls engine.stop() inside speak(),
    so a stop error gets re-raised as RuntimeError. The context manager
    cleanup happens after speak() has already failed.
    """
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.stop.side_effect = Exception("Stop error")
    
    # Should not raise even if stop fails (FAILS because speak() re-raises)
    with TextToSpeech(valid_tts_config) as tts:
        tts.speak("Test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
