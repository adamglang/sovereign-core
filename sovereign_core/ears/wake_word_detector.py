"""
Wake word detector for Sovereign using Picovoice Porcupine.

Continuously listens for "Hey Sovereign" wake word with low CPU usage.
"""

import logging
import struct
import threading
from datetime import datetime
from typing import Iterator, Optional

import pvporcupine
import pyaudio

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


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


class WakeWordDetector:
    """
    Continuously listens for wake word using Picovoice Porcupine.
    
    Runs in a background thread for low CPU usage and non-blocking operation.
    Yields detection events when wake word is heard.
    """
    
    def __init__(self, config: WakeWordConfig):
        """
        Initialize the wake word detector.
        
        Args:
            config: Wake word configuration with access key and model path
        """
        self.config = config
        self._porcupine: Optional[pvporcupine.Porcupine] = None
        self._audio_stream: Optional[pyaudio.Stream] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info("WakeWordDetector initialized")
    
    def start(self) -> Iterator[WakeWordDetection]:
        """
        Start listening for wake word.
        
        This method initializes Porcupine and the audio stream, then continuously
        listens for the wake word in a background thread. It yields detection events
        when the wake word is detected.
        
        Yields:
            WakeWordDetection: Detection events with confidence scores
            
        Raises:
            RuntimeError: If detector is already running or initialization fails
        """
        if self._running:
            raise RuntimeError("Wake word detector is already running")
        
        try:
            # Initialize Porcupine
            logger.info("Initializing Porcupine wake word engine")
            
            porcupine_kwargs = {
                "access_key": self.config.access_key,
                "keywords": ["hey sovereign"],
                "sensitivities": [self.config.sensitivity],
            }
            
            if self.config.model_path:
                porcupine_kwargs["model_path"] = self.config.model_path
            
            self._porcupine = pvporcupine.create(**porcupine_kwargs)
            
            logger.info(
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
            
            logger.info("Audio stream opened")
            
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
                        confidence = self.config.sensitivity
                        
                        logger.info(
                            f"Wake word detected (confidence={confidence:.2f}, "
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
            logger.error(f"Failed to start wake word detector: {e}")
            self.stop()
            raise RuntimeError(f"Wake word detector initialization failed: {e}")
    
    def stop(self) -> None:
        """
        Stop listening for wake word and clean up resources.
        
        This method safely terminates the audio stream and releases Porcupine
        resources. It can be called multiple times safely.
        """
        logger.info("Stopping wake word detector")
        
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
        
        logger.info("Wake word detector stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()