"""Ears module - Wake-word detection and audio capture."""

from .wake_word_interface import WakeWordDetection, WakeWordProvider
from .wake_word_factory import get_wake_word_provider
from .wake_word_detector import WakeWordDetector

__all__ = [
    "WakeWordDetection",
    "WakeWordProvider",
    "get_wake_word_provider",
    "WakeWordDetector",
]