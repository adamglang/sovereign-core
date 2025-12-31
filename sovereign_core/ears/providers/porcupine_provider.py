"""
Porcupine wake word detection provider implementation.

This module implements the WakeWordProvider interface for Picovoice Porcupine,
supporting offline wake word detection with low CPU usage.
"""

import logging
import struct
from datetime import datetime
from typing import Iterator, Optional

import pvporcupine
import pyaudio

from ..wake_word_interface import WakeWordDetection, WakeWordProvider

logger = logging.getLogger(__name__)


class PorcupineProvider(WakeWordProvider):
    """
    Porcupine implementation of the wake word detection provider interface.
    
    Uses Picovoice Porcupine for offline wake word detection with minimal
    CPU usage. Supports custom wake words and sensitivity configuration.
    """
    
    def __init__(
        self,
        access_key: str,
        sensitivity: float,
        keywords: list[str] | None = None,
        keyword_path: str | None = None,
    ):
        """
        Initialize the Porcupine provider.
        
        Args:
            access_key: Picovoice access key for authentication
            sensitivity: Detection sensitivity 0.0-1.0
            keywords: List of built-in wake word keywords (optional if keyword_path provided)
            keyword_path: Optional path to custom .ppn wake word file
        
        Raises:
            ValueError: If access_key is not provided
        """
        if not access_key:
            raise ValueError(
                "Porcupine access_key is required. "
                "Get your key from https://console.picovoice.ai/"
            )
        
        self.access_key = access_key
        self.keywords = keywords
        self.sensitivity = sensitivity
        self.keyword_path = keyword_path
        
        self._porcupine: Optional[pvporcupine.Porcupine] = None
        self._audio_stream: Optional[pyaudio.Stream] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._running = False
        
        if self.keyword_path:
            logger.info(f"Initialized Porcupine provider with custom wake word: {self.keyword_path}")
        else:
            logger.info(f"Initialized Porcupine provider with built-in keywords: {self.keywords}")
    
    def start(self) -> Iterator[WakeWordDetection]:
        """
        Start listening for wake word using Porcupine.
        
        Initializes Porcupine engine and audio stream, then continuously
        listens for the configured wake word(s). Yields detection events
        when a wake word is detected.
        
        Yields:
            WakeWordDetection: Detection events with confidence scores
            
        Raises:
            RuntimeError: If detector is already running or initialization fails
        """
        if self._running:
            raise RuntimeError("Wake word detector is already running")
        
        try:
            # Initialize Porcupine (log at debug level to avoid spam)
            logger.debug("Initializing Porcupine wake word engine")
            
            porcupine_kwargs = {
                "access_key": self.access_key,
            }
            
            # Use custom .ppn file if provided, otherwise use built-in keywords
            if self.keyword_path:
                porcupine_kwargs["keyword_paths"] = [self.keyword_path]
                porcupine_kwargs["sensitivities"] = [self.sensitivity]
            else:
                porcupine_kwargs["keywords"] = self.keywords
                porcupine_kwargs["sensitivities"] = [self.sensitivity] * len(self.keywords)
            
            self._porcupine = pvporcupine.create(**porcupine_kwargs)
            
            logger.debug(
                f"Porcupine initialized (sample_rate={self._porcupine.sample_rate}, "
                f"frame_length={self._porcupine.frame_length})"
            )
            
            # Initialize PyAudio
            self._pa = pyaudio.PyAudio()
            
            self._audio_stream = self._pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
            )
            
            logger.info("Porcupine wake word detector started and listening")
            
            self._running = True
            
            # Yield detection events as they occur
            while self._running:
                try:
                    pcm = self._audio_stream.read(
                        self._porcupine.frame_length,
                        exception_on_overflow=False,
                    )
                    pcm = struct.unpack_from(
                        "h" * self._porcupine.frame_length,
                        pcm,
                    )
                    
                    keyword_index = self._porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        timestamp = datetime.now()
                        confidence = self.sensitivity
                        
                        # Get keyword name for logging
                        if self.keyword_path:
                            keyword_name = f"custom wake word ({self.keyword_path})"
                        else:
                            keyword_name = self.keywords[keyword_index]
                        
                        logger.info(
                            f"Wake word detected (keyword={keyword_name}, "
                            f"confidence={confidence:.2f}, "
                            f"timestamp={timestamp.isoformat()})"
                        )
                        
                        yield WakeWordDetection(
                            detected=True,
                            confidence=confidence,
                            timestamp=timestamp,
                        )
                
                except Exception as e:
                    if self._running:
                        logger.error(f"Error processing audio frame: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Failed to start Porcupine wake word detector: {e}")
            self.stop()
            raise RuntimeError(f"Wake word detector initialization failed: {e}")
    
    def stop(self) -> None:
        """
        Stop listening for wake word and clean up Porcupine resources.
        
        Safely terminates the audio stream and releases Porcupine engine
        resources. Can be called multiple times safely.
        """
        logger.info("Stopping Porcupine wake word detector")
        
        self._running = False
        
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                logger.debug("Audio stream closed")
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            finally:
                self._audio_stream = None
        
        if self._pa is not None:
            try:
                self._pa.terminate()
                logger.debug("PyAudio terminated")
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
            finally:
                self._pa = None
        
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
                logger.debug("Porcupine engine deleted")
            except Exception as e:
                logger.warning(f"Error deleting Porcupine instance: {e}")
            finally:
                self._porcupine = None
        
        logger.info("Porcupine wake word detector stopped")