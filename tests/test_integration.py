"""Integration tests for the full SovereignCore pipeline."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock modules before importing to avoid dependency issues
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['pvporcupine'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
sys.modules['pyttsx3'] = MagicMock()
sys.modules['faster_whisper'] = MagicMock()
sys.modules['sounddevice'] = MagicMock()

from sovereign_core.main import SovereignCore
from sovereign_core.router.models import IntentType, RouterResponse


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock required environment variables for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "test-porcupine-key")


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    log_file = tmp_path / "test.log"
    db_file = tmp_path / "test_sovereign.db"
    
    config_content = f"""
logging:
  level: INFO
  file: {str(log_file).replace(chr(92), '/')}
  console: false

wake_word:
  access_key: test-access-key
  model_path: null
  sensitivity: 0.7

audio:
  sample_rate: 16000
  channels: 1
  frame_duration_ms: 30
  device_index: null

stt:
  model_size: base
  device: cpu
  compute_type: int8
  model_dir: ./models/whisper

llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7
  max_tokens: 1000

tts:
  provider: windows
  voice: null
  rate: 1.0

ipc:
  database_path: {str(db_file).replace(chr(92), '/')}

router:
  action_keywords:
    - play
    - pause
    - stop
    - resume
    - next
    - previous
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    
    yield str(config_file)


@pytest.fixture(autouse=True)
def mock_all_components():
    """Mock all SovereignCore components."""
    with patch('sovereign_core.main.WakeWordDetector') as mock_wake_class, \
         patch('sovereign_core.main.AudioCapture') as mock_audio_class, \
         patch('sovereign_core.main.SpeechToText') as mock_stt_class, \
         patch('sovereign_core.main.TextToSpeech') as mock_tts_class, \
         patch('sovereign_core.main.Router') as mock_router_class, \
         patch('sovereign_core.main.init_database') as mock_init_db, \
         patch('sovereign_core.main.get_llm_provider') as mock_get_llm:
        
        # Create mock instances
        mock_wake = MagicMock()
        mock_audio = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()
        mock_router = MagicMock()
        mock_llm = MagicMock()
        
        # Configure returns
        mock_wake_class.return_value = mock_wake
        mock_audio_class.return_value = mock_audio
        mock_stt_class.return_value = mock_stt
        mock_tts_class.return_value = mock_tts
        mock_router_class.return_value = mock_router
        mock_get_llm.return_value = mock_llm
        
        # Set default behaviors
        mock_wake.wait_for_wake_word.return_value = False
        mock_wake.cleanup.return_value = None
        mock_audio.capture_audio.return_value = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        mock_audio.cleanup.return_value = None
        mock_stt.transcribe.return_value = "default question"
        mock_stt.cleanup.return_value = None
        mock_tts.speak.return_value = None
        mock_router.route.return_value = RouterResponse(
            type=IntentType.CONVERSATIONAL,
            conversational_response="default response"
        )
        
        yield {
            'wake_word_detector': mock_wake,
            'audio_capture': mock_audio,
            'stt': mock_stt,
            'tts': mock_tts,
            'router': mock_router,
            'ipc_init': mock_init_db,
            'llm_provider': mock_llm,
        }


def test_sovereign_core_initialization(temp_config_file, mock_all_components):
    """Test SovereignCore initializes all components in correct order."""
    core = SovereignCore(config_path=temp_config_file)
    
    # Verify all components were initialized
    assert core.wake_word_detector is not None
    assert core.audio_capture is not None
    assert core.stt is not None
    assert core.tts is not None
    assert core.router is not None
    
    # Verify IPC database was initialized first
    mock_all_components['ipc_init'].assert_called_once()
    
    # Verify conversation history is initialized
    assert core.conversation_history == []
    assert core.running is False


def test_full_conversational_flow(temp_config_file, mock_all_components):
    """Test happy path: wake word → audio → transcription → LLM → TTS."""
    # Setup: wake word detection returns True once, then False to stop loop
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, False]
    
    # Setup: STT returns user question
    mock_all_components['stt'].transcribe.return_value = "What is the weather today?"
    
    # Setup: Router returns conversational response
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="The weather is sunny and 72 degrees.",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify flow: wake word → audio capture → transcription
    mock_all_components['wake_word_detector'].wait_for_wake_word.assert_called()
    mock_all_components['audio_capture'].capture_audio.assert_called_once()
    mock_all_components['stt'].transcribe.assert_called_once()
    
    # Verify router received transcribed text
    mock_all_components['router'].route.assert_called_once()
    call_args = mock_all_components['router'].route.call_args
    assert call_args[1]['utterance'] == "What is the weather today?"
    
    # Verify conversation history was passed to router
    assert 'conversation_history' in call_args[1]
    
    # Verify TTS spoke the response
    mock_all_components['tts'].speak.assert_called_once_with("The weather is sunny and 72 degrees.")
    
    # Verify conversation history was updated
    assert len(core.conversation_history) == 2  # User + Assistant
    assert core.conversation_history[0]['role'] == 'user'
    assert core.conversation_history[0]['content'] == "What is the weather today?"
    assert core.conversation_history[1]['role'] == 'assistant'
    assert core.conversation_history[1]['content'] == "The weather is sunny and 72 degrees."


def test_action_intent_flow(temp_config_file, mock_all_components):
    """Test action intent: user requests action → command written to IPC."""
    # Setup: wake word detection returns True once, then False to stop loop
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, False]
    
    # Setup: STT returns action request
    mock_all_components['stt'].transcribe.return_value = "Play 46 & 2 by Tool"
    
    # Setup: Router returns action intent
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.ACTION,
        action_intent={
            "action": "spotify.play_query",
            "params": {"query": "46 & 2 by Tool"}
        },
        command_id=123,
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify router was called with action request
    mock_all_components['router'].route.assert_called_once()
    call_args = mock_all_components['router'].route.call_args
    assert call_args[1]['utterance'] == "Play 46 & 2 by Tool"
    
    # Verify TTS acknowledged the action
    mock_all_components['tts'].speak.assert_called_once()
    spoken_text = mock_all_components['tts'].speak.call_args[0][0]
    assert "queued" in spoken_text.lower()
    assert "123" in spoken_text


def test_clarification_flow(temp_config_file, mock_all_components):
    """Test clarification: ambiguous request → asks clarifying question."""
    # Setup: wake word detection returns True once, then False to stop loop
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, False]
    
    # Setup: STT returns ambiguous request
    mock_all_components['stt'].transcribe.return_value = "Play something"
    
    # Setup: Router requests clarification
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CLARIFICATION_NEEDED,
        clarification_question="What would you like me to play?",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify TTS asked clarification question
    mock_all_components['tts'].speak.assert_called_once_with("What would you like me to play?")
    
    # Verify conversation history includes user request and clarification
    assert len(core.conversation_history) == 2


def test_conversation_history_management(temp_config_file, mock_all_components):
    """Test conversation history is updated and limited to 20 messages."""
    # Setup: simulate 15 exchanges (30 messages)
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True] * 15 + [False]
    
    mock_all_components['stt'].transcribe.return_value = "Test question"
    
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="Test response",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # History should be limited to 20 messages (10 most recent exchanges)
    assert len(core.conversation_history) == 20
    
    # Verify most recent messages are kept
    assert core.conversation_history[-2]['role'] == 'user'
    assert core.conversation_history[-1]['role'] == 'assistant'


def test_empty_transcription_handling(temp_config_file, mock_all_components):
    """Test error recovery: empty transcription → informs user and continues."""
    # Setup: wake word detected twice, first has empty transcription
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, True, False]
    
    # First transcription is empty, second is valid
    mock_all_components['stt'].transcribe.side_effect = ["", "Valid question"]
    
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="Valid response",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify TTS informed user about empty transcription
    tts_calls = mock_all_components['tts'].speak.call_args_list
    assert len(tts_calls) == 2
    assert "couldn't understand" in tts_calls[0][0][0].lower()
    
    # Verify router was only called once (for valid transcription)
    mock_all_components['router'].route.assert_called_once()


def test_no_audio_captured_handling(temp_config_file, mock_all_components):
    """Test error recovery: no audio captured → informs user and continues."""
    # Setup: wake word detected twice, first has no audio
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, True, False]
    
    # First capture returns None/empty, second is valid
    audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
    mock_all_components['audio_capture'].capture_audio.side_effect = [None, audio_data]
    
    mock_all_components['stt'].transcribe.return_value = "Valid question"
    
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="Valid response",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify TTS informed user about no audio
    tts_calls = mock_all_components['tts'].speak.call_args_list
    assert len(tts_calls) == 2
    assert "didn't hear" in tts_calls[0][0][0].lower()


def test_error_in_main_loop_recovery(temp_config_file, mock_all_components):
    """Test error recovery: component failure → system continues."""
    # Setup: first iteration fails, second succeeds
    mock_all_components['wake_word_detector'].wait_for_wake_word.side_effect = [True, True, False]
    
    audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
    mock_all_components['audio_capture'].capture_audio.return_value = audio_data
    
    # First transcription fails, second succeeds
    mock_all_components['stt'].transcribe.side_effect = [
        RuntimeError("Transcription failed"),
        "Valid question"
    ]
    
    mock_all_components['router'].route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="Valid response",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify system recovered and continued
    # TTS should have been called twice: error message + valid response
    assert mock_all_components['tts'].speak.call_count >= 1
    
    # Router should have been called once (after recovery)
    mock_all_components['router'].route.assert_called_once()


def test_component_cleanup_on_shutdown(temp_config_file, mock_all_components):
    """Test proper cleanup: components cleaned up in reverse order."""
    core = SovereignCore(config_path=temp_config_file)
    core.running = True
    core.stop()
    
    # Verify cleanup was called on all components
    mock_all_components['wake_word_detector'].cleanup.assert_called_once()
    mock_all_components['audio_capture'].cleanup.assert_called_once()
    mock_all_components['stt'].cleanup.assert_called_once()
    
    # Verify running flag is set to False
    assert core.running is False


def test_idempotent_stop(temp_config_file, mock_all_components):
    """Test stop() is idempotent and can be called multiple times."""
    core = SovereignCore(config_path=temp_config_file)
    core.running = True
    
    # Call stop multiple times
    core.stop()
    core.stop()
    core.stop()
    
    # Should not raise any errors
    assert core.running is False


def test_cleanup_errors_handled_gracefully(
    temp_config_file,
    mock_wake_word_detector,
    mock_audio_capture,
    mock_stt,
    mock_tts,
    mock_router,
    mock_ipc_database,
    mock_llm_provider,
):
    """Test cleanup errors don't prevent shutdown."""
    # Setup: cleanup methods raise errors
    mock_wake_word_detector.cleanup.side_effect = Exception("Cleanup error")
    mock_audio_capture.cleanup.side_effect = Exception("Cleanup error")
    mock_stt.cleanup.side_effect = Exception("Cleanup error")
    
    core = SovereignCore(config_path=temp_config_file)
    core.running = True
    
    # Should not raise despite cleanup errors
    core.stop()
    
    assert core.running is False


def test_wake_word_not_detected_continues_listening(
    temp_config_file,
    mock_wake_word_detector,
    mock_audio_capture,
    mock_stt,
    mock_tts,
    mock_router,
    mock_ipc_database,
    mock_llm_provider,
):
    """Test system continues listening when wake word not detected."""
    # Setup: wake word not detected twice, then detected, then stop
    mock_wake_word_detector.wait_for_wake_word.side_effect = [False, False, True, False]
    
    mock_stt.transcribe.return_value = "Test question"
    mock_router.route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response="Test response",
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify wake word detector was called 4 times (2 false, 1 true, 1 false to stop)
    assert mock_wake_word_detector.wait_for_wake_word.call_count == 4
    
    # Verify audio capture only called once (when wake word detected)
    mock_audio_capture.capture_audio.assert_called_once()


def test_full_pipeline_data_flow(
    temp_config_file,
    mock_wake_word_detector,
    mock_audio_capture,
    mock_stt,
    mock_tts,
    mock_router,
    mock_ipc_database,
    mock_llm_provider,
):
    """Test data flows correctly through entire pipeline."""
    mock_wake_word_detector.wait_for_wake_word.side_effect = [True, False]
    
    # Setup audio data
    audio_data = np.random.randint(-32768, 32767, size=48000, dtype=np.int16)
    mock_audio_capture.capture_audio.return_value = audio_data
    
    # Setup transcription
    transcribed_text = "What is the meaning of life?"
    mock_stt.transcribe.return_value = transcribed_text
    
    # Setup router response
    llm_response = "The meaning of life is to find purpose and happiness."
    mock_router.route.return_value = RouterResponse(
        type=IntentType.CONVERSATIONAL,
        conversational_response=llm_response,
    )
    
    core = SovereignCore(config_path=temp_config_file)
    core.main_loop()
    
    # Verify data flow:
    # 1. Audio data passed to STT
    stt_call_args = mock_stt.transcribe.call_args[0][0]
    assert isinstance(stt_call_args, (np.ndarray, type(audio_data)))
    
    # 2. Transcribed text passed to router
    router_call_args = mock_router.route.call_args[1]
    assert router_call_args['utterance'] == transcribed_text
    
    # 3. Router response passed to TTS
    tts_call_args = mock_tts.speak.call_args[0][0]
    assert tts_call_args == llm_response
    
    # 4. Conversation history updated correctly
    assert core.conversation_history[0]['content'] == transcribed_text
    assert core.conversation_history[1]['content'] == llm_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])