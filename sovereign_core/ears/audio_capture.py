"""
Audio capture module for Sovereign.

Records audio after wake word detection for speech-to-text processing.
"""

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

logger = logging.getLogger(__name__)


class AudioRecording:
    """Audio recording result with metadata."""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        """
        Initialize an audio recording.
        
        Args:
            audio_data: Raw audio samples as numpy array
            sample_rate: Sample rate used for recording
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate


class AudioCapture:
    """
    Captures audio from microphone for speech-to-text processing.
    
    Uses sounddevice library to record audio after wake word detection.
    Supports configurable duration and audio parameters from config.
    """
    
    def __init__(self, config: AudioConfig, default_duration: float = 5.0):
        """
        Initialize the audio capture.
        
        Args:
            config: Audio configuration with sample rate, channels, and device
            default_duration: Default recording duration in seconds (default: 5.0)
        """
        self.config = config
        self.default_duration = default_duration
        
        logger.info(
            f"AudioCapture initialized (sample_rate={config.sample_rate}, "
            f"channels={config.channels}, default_duration={default_duration}s)"
        )
    
    def capture(self, duration: Optional[float] = None) -> AudioRecording:
        """
        Record audio from microphone for specified duration.
        
        Records audio using sounddevice and returns it as a numpy array
        suitable for faster-whisper processing (16-bit PCM, mono).
        
        Args:
            duration: Recording duration in seconds. If None, uses default_duration
        
        Returns:
            AudioRecording: Captured audio data with sample rate metadata
        
        Raises:
            RuntimeError: If microphone access fails or recording fails
        """
        if duration is None:
            duration = self.default_duration
        
        try:
            logger.info(f"Starting audio capture for {duration}s")
            
            # Record audio using sounddevice
            audio_data = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.int16,
                device=self.config.device_index,
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Convert to mono if needed (take first channel)
            if self.config.channels > 1:
                audio_data = audio_data[:, 0]
            else:
                audio_data = audio_data.flatten()
            
            logger.info(
                f"Audio captured successfully (samples={len(audio_data)}, "
                f"duration={duration}s)"
            )
            
            return AudioRecording(
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
            )
        
        except sd.PortAudioError as e:
            logger.error(f"Microphone access failed: {e}")
            raise RuntimeError(f"Failed to access microphone: {e}")
        
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            raise RuntimeError(f"Audio capture failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        logger.debug("AudioCapture context exited")