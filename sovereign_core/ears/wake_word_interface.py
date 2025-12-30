"""
Abstract base class for wake word detection providers.

This module defines the interface that all wake word detection providers must implement,
enabling zero-friction wake word engine swapping through configuration changes only.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator


class WakeWordDetection:
    """Wake word detection event."""
    
    def __init__(self, detected: bool, confidence: float, timestamp: datetime):
        """
        Initialize a wake word detection event.
        
        Args:
            detected: Whether the wake word was detected
            confidence: Confidence score (0.0 to 1.0)
            timestamp: When the detection occurred
        """
        self.detected = detected
        self.confidence = confidence
        self.timestamp = timestamp


class WakeWordProvider(ABC):
    """
    Abstract base class for wake word detection providers.
    
    All provider implementations must inherit from this class and implement
    the required methods. This ensures consistent behavior across different
    wake word engines (Porcupine, Snowboy, Mycroft Precise, etc.).
    """
    
    @abstractmethod
    def start(self) -> Iterator[WakeWordDetection]:
        """
        Start listening for wake word.
        
        This method initializes the wake word engine and audio stream, then
        continuously listens for the wake word. It yields detection events
        when the wake word is detected.
        
        Yields:
            WakeWordDetection: Detection events with confidence scores
            
        Raises:
            RuntimeError: If detector is already running or initialization fails
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop listening for wake word and clean up resources.
        
        This method safely terminates the audio stream and releases engine
        resources. It can be called multiple times safely.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()