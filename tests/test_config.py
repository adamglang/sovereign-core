"""Test configuration loading functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from sovereign_core.config import (
    AudioConfig,
    ConversationConfig,
    load_config,
    LLMConfig,
    LoggingConfig,
    SovereignConfig,
    STTConfig,
    TTSConfig,
    WakeWordConfig,
    _resolve_env_vars,
)


@pytest.fixture
def valid_config_data():
    """Valid configuration data for testing."""
    return {
        "wake_word": {
            "access_key": "${PORCUPINE_ACCESS_KEY}",
            "sensitivity": 0.7,
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
        },
        "turn_taking": {
            "vad_aggressiveness": 2,
            "min_speech_duration_ms": 300,
            "end_silence_duration_ms": 700,
            "post_speech_grace_ms": 500,
            "max_recording_duration_s": 15,
            "vad_frame_duration_ms": 30,
        },
        "stt": {
            "model_size": "base",
            "device": "cpu",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
        "tts": {
            "voice_model": "en_US-lessac-medium",
            "speaker_id": None,
            "use_cuda": False,
        },
        "ipc": {
            "database_path": "./test.db",
        },
        "router": {
            "action_keywords": ["play", "pause", "stop"],
        },
        "conversation": {
            "max_history_messages": 10,
            "context_messages": 10,
            "follow_up_timeout_seconds": 10.0,
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/test.log",
        },
    }


@pytest.fixture
def temp_config_file(valid_config_data):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(valid_config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink(missing_ok=True)


def test_successful_config_loading(temp_config_file):
    """Test successful configuration loading with all required env vars."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "PORCUPINE_ACCESS_KEY": "test-porcupine-key",
    }):
        config = load_config(temp_config_file)
        
        assert isinstance(config, SovereignConfig)
        assert config.wake_word.access_key == "test-porcupine-key"
        assert config.wake_word.sensitivity == 0.7
        assert config.audio.sample_rate == 16000
        assert config.stt.model_size == "base"
        assert config.llm.provider == "openai"
        assert config.tts.voice_model == "en_US-lessac-medium"
        assert config.tts.use_cuda is False
        assert config.logging.level == "INFO"


def test_missing_openai_api_key(temp_config_file):
    """Test error when OPENAI_API_KEY is missing."""
    with patch.dict(os.environ, {
        "PORCUPINE_ACCESS_KEY": "test-porcupine-key",
    }, clear=True):
        with pytest.raises(ValueError) as exc_info:
            load_config(temp_config_file)
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "Required environment variables" in str(exc_info.value)


def test_missing_porcupine_access_key(temp_config_file):
    """Test error when PORCUPINE_ACCESS_KEY is missing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
    }, clear=True):
        with pytest.raises(ValueError) as exc_info:
            load_config(temp_config_file)
        
        assert "PORCUPINE_ACCESS_KEY" in str(exc_info.value)
        assert "Required environment variables" in str(exc_info.value)


def test_missing_both_api_keys(temp_config_file):
    """Test error when both required env vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            load_config(temp_config_file)
        
        error_msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_msg
        assert "PORCUPINE_ACCESS_KEY" in error_msg


def test_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config("nonexistent_config.yaml")
    
    assert "Configuration file not found" in str(exc_info.value)


def test_invalid_yaml():
    """Test error with malformed YAML."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("invalid: yaml: content:\n  - bad indentation\nmissing quote")
        temp_path = f.name
    
    try:
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "PORCUPINE_ACCESS_KEY": "test-key",
        }):
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_env_var_substitution():
    """Test environment variable substitution in config."""
    test_data = {
        "simple": "${TEST_VAR}",
        "nested": {
            "value": "${NESTED_VAR}",
            "plain": "no substitution",
        },
        "list": ["${LIST_VAR}", "plain"],
    }
    
    with patch.dict(os.environ, {
        "TEST_VAR": "test_value",
        "NESTED_VAR": "nested_value",
        "LIST_VAR": "list_value",
    }):
        result = _resolve_env_vars(test_data)
        
        assert result["simple"] == "test_value"
        assert result["nested"]["value"] == "nested_value"
        assert result["nested"]["plain"] == "no substitution"
        assert result["list"][0] == "list_value"
        assert result["list"][1] == "plain"


def test_env_var_substitution_missing_var():
    """Test env var substitution when variable doesn't exist."""
    test_data = {"key": "${MISSING_VAR}"}
    
    with patch.dict(os.environ, {}, clear=True):
        result = _resolve_env_vars(test_data)
        # Should keep the placeholder if env var not found
        assert result["key"] == "${MISSING_VAR}"


def test_wake_word_config_validation():
    """Test WakeWordConfig validation."""
    # Valid config
    config = WakeWordConfig(access_key="test-key", sensitivity=0.5)
    assert config.sensitivity == 0.5
    
    # Invalid sensitivity (too high)
    with pytest.raises(ValidationError):
        WakeWordConfig(access_key="test-key", sensitivity=1.5)
    
    # Invalid sensitivity (negative)
    with pytest.raises(ValidationError):
        WakeWordConfig(access_key="test-key", sensitivity=-0.1)


def test_logging_config_validation():
    """Test LoggingConfig validation."""
    # Valid levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config = LoggingConfig(level=level, file="test.log")
        assert config.level == level
    
    # Case-insensitive
    config = LoggingConfig(level="info", file="test.log")
    assert config.level == "INFO"
    
    # Invalid level
    with pytest.raises(ValidationError) as exc_info:
        LoggingConfig(level="INVALID", file="test.log")
    assert "Invalid log level" in str(exc_info.value)


def test_llm_config_defaults():
    """Test LLMConfig default values."""
    config = LLMConfig()
    
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000


def test_tts_config_validation():
    """Test TTSConfig validation."""
    # Valid config with defaults
    config = TTSConfig()
    assert config.voice_model == "en_US-lessac-medium"
    assert config.speaker_id is None
    assert config.use_cuda is True
    
    # Valid config with custom values
    config = TTSConfig(voice_model="en_US-amy-medium", speaker_id=42, use_cuda=False)
    assert config.voice_model == "en_US-amy-medium"
    assert config.speaker_id == 42
    assert config.use_cuda is False


def test_config_uses_sovereign_config_path_env_var():
    """Test that load_config respects SOVEREIGN_CONFIG_PATH env var."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump({
            "wake_word": {"access_key": "${PORCUPINE_ACCESS_KEY}"},
            "audio": {},
            "turn_taking": {},
            "stt": {},
            "llm": {},
            "tts": {},
            "ipc": {},
            "router": {},
            "conversation": {},
            "logging": {},
        }, f)
        temp_path = f.name
    
    try:
        with patch.dict(os.environ, {
            "SOVEREIGN_CONFIG_PATH": temp_path,
            "OPENAI_API_KEY": "test-key",
            "PORCUPINE_ACCESS_KEY": "test-key",
        }):
            config = load_config()  # No explicit path
            assert isinstance(config, SovereignConfig)
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])