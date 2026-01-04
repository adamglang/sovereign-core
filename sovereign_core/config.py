"""
Configuration loader for Sovereign Core.

Loads settings from config.yaml and merges environment variables for API keys.
"""

import os
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


class WakeWordConfig(BaseModel):
    """Wake word detection configuration."""

    access_key: str
    keywords: list[str] = Field(default=["hey sovereign"])
    keyword_path: Optional[str] = None
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 30
    device_index: Optional[int] = None


class TurnTakingConfig(BaseModel):
    """Turn-taking and voice activity detection configuration."""
    
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    min_speech_duration_ms: int = Field(default=300, ge=100)
    end_silence_duration_ms: int = Field(default=700, ge=200)
    post_speech_grace_ms: int = Field(default=500, ge=0)
    max_recording_duration_s: int = Field(default=15, ge=5)
    vad_frame_duration_ms: int = Field(default=30)
    
    @field_validator("vad_frame_duration_ms")
    @classmethod
    def validate_frame_duration(cls, v: int) -> int:
        if v not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")
        return v


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""

    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    model_dir: str = "./models/whisper"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = 1000


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    # Piper voice model name (without .onnx extension)
    voice_model: str = "en_US-lessac-medium"
    
    # Optional speaker ID for multi-speaker models (None = default speaker)
    speaker_id: Optional[int] = None
    
    # GPU acceleration via ONNX Runtime (requires CUDA on RTX 4070)
    use_cuda: bool = Field(default=True)


class IPCConfig(BaseModel):
    """Inter-process communication configuration."""

    database_path: str = "./sovereign.db"
    timeout: int = 30


class RouterConfig(BaseModel):
    """Router configuration."""

    action_keywords: list[str] = Field(default_factory=list)


class ConversationConfig(BaseModel):
    """Conversation history configuration."""

    max_history_messages: int = Field(default=10, ge=1)
    context_messages: int = Field(default=10, ge=1)
    follow_up_timeout_seconds: float = Field(default=10.0, ge=0.0)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "./logs/sovereign.log"
    console: bool = True
    max_bytes: int = Field(default=10_485_760)
    backup_count: int = Field(default=5)
    third_party_level: str = Field(default="WARNING")
    third_party_loggers: list[str] = Field(default_factory=list)

    @field_validator("level", "third_party_level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class SovereignConfig(BaseModel):
    """Complete Sovereign configuration."""

    wake_word: WakeWordConfig
    audio: AudioConfig
    turn_taking: TurnTakingConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    ipc: IPCConfig
    router: RouterConfig
    conversation: ConversationConfig
    logging: LoggingConfig


def _resolve_env_vars(data: Any) -> Any:
    """
    Recursively resolve environment variable placeholders in config data.
    
    Replaces ${VAR_NAME} with the value from os.environ.
    
    Args:
        data: Configuration data (dict, list, or primitive)
    
    Returns:
        Data with environment variables resolved
    """
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.getenv(env_var, data)
    return data


def load_config(config_path: Optional[str] = None) -> SovereignConfig:
    """
    Load and validate Sovereign configuration.
    
    Reads from SOVEREIGN_CONFIG_PATH environment variable or the provided path,
    defaulting to 'config.yaml'. Validates using Pydantic models and merges
    required environment variables.
    
    Args:
        config_path: Optional path to config file. If not provided, uses
                    SOVEREIGN_CONFIG_PATH env var or defaults to 'config.yaml'
    
    Returns:
        SovereignConfig: Validated configuration instance
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required environment variables are missing
        yaml.YAMLError: If config file is invalid YAML
        pydantic.ValidationError: If config doesn't match schema
    """
    if config_path is None:
        config_path = os.getenv("SOVEREIGN_CONFIG_PATH", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Please ensure the file exists or set SOVEREIGN_CONFIG_PATH correctly."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    config_data = _resolve_env_vars(raw_config)
    
    required_env_vars = {
        "OPENAI_API_KEY": "Get your API key from https://platform.openai.com/api-keys",
        "PORCUPINE_ACCESS_KEY": "Get your access key from https://console.picovoice.ai/",
    }
    
    missing_vars = []
    for var_name, help_text in required_env_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"  - {var_name}: {help_text}")
    
    if missing_vars:
        raise ValueError(
            "Required environment variables are not set:\n" + "\n".join(missing_vars)
        )
    
    config = SovereignConfig(**config_data)
    
    return config