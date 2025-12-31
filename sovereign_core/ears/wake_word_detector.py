"""
Wake word detector for Sovereign.

This module provides a backward-compatible wrapper around the wake word provider
abstraction. It delegates to the configured provider (default: Porcupine) while
maintaining the existing API for seamless migration.

For new code, prefer using the factory directly:
    from sovereign_core.ears.wake_word_factory import get_wake_word_provider
    provider = get_wake_word_provider("porcupine", config)
"""

import logging
from typing import Iterator

from ..config import WakeWordConfig
from .wake_word_factory import get_wake_word_provider
from .wake_word_interface import WakeWordDetection, WakeWordProvider

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Backward-compatible wake word detector wrapper.
    
    This class maintains the original API while delegating to the pluggable
    provider system. It automatically selects the appropriate provider based
    on configuration.
    """
    
    def __init__(
        self,
        config: WakeWordConfig = None,
        provider_name: str = "porcupine",
        access_key: str = None,
        model_path: str = None,
        sensitivity: float = 0.5,
    ):
        """
        Initialize the wake word detector.
        
        Args:
            config: Wake word configuration with access key and model path (preferred)
            provider_name: Name of the wake word provider to use (default: "porcupine")
            access_key: Porcupine access key (alternative to config)
            model_path: Optional path to custom model (alternative to config)
            sensitivity: Detection sensitivity 0.0-1.0 (alternative to config)
        """
        # Support both new config-based and old parameter-based initialization
        if config is not None:
            self.config = config
            provider_config = {
                "access_key": config.access_key,
                "keywords": config.keywords,
                "sensitivity": config.sensitivity,
            }
            if hasattr(config, 'keyword_path') and config.keyword_path:
                provider_config["keyword_path"] = config.keyword_path
        else:
            # Backward compatibility: build config from individual parameters
            from ..config import WakeWordConfig as WakeWordConfigClass
            self.config = WakeWordConfigClass(
                access_key=access_key,
                keyword_path=model_path,
                sensitivity=sensitivity,
            )
            provider_config = {
                "access_key": access_key,
                "keywords": self.config.keywords,
                "sensitivity": sensitivity,
            }
            if model_path:
                provider_config["keyword_path"] = model_path
        
        self.provider_name = provider_name
        
        # Create the underlying provider
        self._provider: WakeWordProvider = get_wake_word_provider(
            provider_name=provider_name,
            config=provider_config,
        )
        
        self._detection_generator = None
        
        logger.info(
            f"WakeWordDetector initialized with {provider_name} provider"
        )
    
    def start(self) -> Iterator[WakeWordDetection]:
        """
        Start listening for wake word.
        
        Delegates to the underlying provider's start method. Yields detection
        events when the wake word is detected.
        
        Yields:
            WakeWordDetection: Detection events with confidence scores
            
        Raises:
            RuntimeError: If detector is already running or initialization fails
        """
        self._detection_generator = self._provider.start()
        return self._detection_generator
    
    def wait_for_wake_word(self) -> bool:
        """
        Wait for a single wake word detection (blocking).
        
        Backward-compatible method that blocks until a wake word is detected.
        
        Returns:
            bool: True if wake word detected, False otherwise
        """
        try:
            if self._detection_generator is None:
                self._detection_generator = self._provider.start()
            
            detection = next(self._detection_generator)
            return detection.detected
        except StopIteration:
            logger.warning("Wake word detector generator stopped unexpectedly")
            self._detection_generator = None
            return False
        except Exception as e:
            logger.error(f"Error waiting for wake word: {e}", exc_info=True)
            self._detection_generator = None
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources (backward-compatible alias for stop).
        """
        self.stop()
    
    def stop(self) -> None:
        """
        Stop listening for wake word and clean up resources.
        
        Delegates to the underlying provider's stop method.
        """
        self._provider.stop()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()