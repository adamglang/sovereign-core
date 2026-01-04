"""
Audio capture module for Sovereign.

Records audio after wake word detection for speech-to-text processing.
"""

import logging
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig, TurnTakingConfig

# Optional webrtcvad import for testing without C++ build tools
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    webrtcvad = None  # type: ignore
    WEBRTCVAD_AVAILABLE = False

logger = logging.getLogger(__name__)


class TurnTakingState(Enum):
    """State machine for turn-taking detection."""
    WAITING_FOR_SPEECH = auto()
    SPEECH_DETECTED = auto()
    SILENCE_AFTER_SPEECH = auto()
    GRACE_PERIOD = auto()
    FINALIZED = auto()


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
    
    def __init__(
        self,
        config: AudioConfig,
        turn_taking_config: Optional[TurnTakingConfig] = None,
        default_duration: float = 5.0,
    ):
        """
        Initialize the audio capture.
        
        Args:
            config: Audio configuration with sample rate, channels, and device
            turn_taking_config: Turn-taking configuration with VAD settings.
                              If None, VAD-based capture is disabled.
            default_duration: Default recording duration in seconds (default: 5.0)
        """
        self.config = config
        self.turn_taking_config = turn_taking_config
        self.default_duration = default_duration
        
        # Initialize VAD if turn-taking is enabled
        if turn_taking_config is not None:
            if not WEBRTCVAD_AVAILABLE or webrtcvad is None:
                raise RuntimeError(
                    "webrtcvad is required for VAD-based turn-taking but is not installed. "
                    "Install it with: pip install webrtcvad"
                )
            self.vad = webrtcvad.Vad(turn_taking_config.vad_aggressiveness)
            logger.info(
                f"AudioCapture initialized with VAD (sample_rate={config.sample_rate}, "
                f"channels={config.channels}, vad_aggressiveness={turn_taking_config.vad_aggressiveness})"
            )
        else:
            self.vad = None
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
        
        except Exception as e:
            # Check if it's a PortAudioError (sounddevice specific)
            if type(e).__name__ == 'PortAudioError':
                logger.error(f"Microphone access failed: {e}")
                raise RuntimeError(f"Failed to access microphone: {e}")
            else:
                logger.error(f"Audio capture failed: {e}")
                raise RuntimeError(f"Audio capture failed: {e}")
    
    def capture_with_vad(self) -> AudioRecording:
        """
        Record audio using VAD-based dynamic endpointing with turn-taking state machine.
        
        Implements intelligent turn-taking detection:
        1. WAITING_FOR_SPEECH - Buffer frames until VAD detects speech
        2. SPEECH_DETECTED - Actively recording speech, reset silence counters
        3. SILENCE_AFTER_SPEECH - Detected silence, accumulate silence duration
        4. GRACE_PERIOD - Silence threshold met, grace period active (cancellable if speech resumes)
        5. FINALIZED - Grace period complete or max duration reached
        
        Returns:
            AudioRecording: Captured audio data with all buffered frames
        
        Raises:
            RuntimeError: If VAD is not initialized or recording fails
        """
        if self.turn_taking_config is None or self.vad is None:
            raise RuntimeError(
                "VAD-based capture requires turn_taking_config to be provided during initialization"
            )
        
        try:
            # Calculate frame parameters from config
            frame_duration_ms = self.turn_taking_config.vad_frame_duration_ms
            frame_samples = (self.config.sample_rate * frame_duration_ms) // 1000
            
            # Initialize state machine
            state = TurnTakingState.WAITING_FOR_SPEECH
            audio_buffer = []
            
            # Timing trackers (in milliseconds)
            total_duration_ms = 0
            speech_duration_ms = 0
            silence_duration_ms = 0
            
            # Open audio stream for frame-by-frame processing
            stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.int16,
                device=self.config.device_index,
                blocksize=frame_samples,
            )
            
            stream.start()
            
            logger.info(
                "VAD-based capture started",
                extra={
                    "event": "capture_start",
                    "state": state.name,
                    "frame_duration_ms": frame_duration_ms,
                    "frame_samples": frame_samples,
                }
            )
            
            while state != TurnTakingState.FINALIZED:
                # Read one frame
                frame_data, overflowed = stream.read(frame_samples)
                
                if overflowed:
                    logger.warning("Audio buffer overflow detected")
                
                # Convert to mono if needed
                if self.config.channels > 1:
                    frame_mono = frame_data[:, 0]
                else:
                    frame_mono = frame_data.flatten()
                
                # Buffer the frame
                audio_buffer.append(frame_mono)
                total_duration_ms += frame_duration_ms
                
                # Convert frame to bytes for VAD (16-bit PCM)
                frame_bytes = frame_mono.astype(np.int16).tobytes()
                
                # Run VAD on frame
                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.config.sample_rate)
                except Exception as e:
                    logger.error(f"VAD processing failed: {e}", extra={"error": str(e)})
                    # Fallback: assume silence to avoid infinite recording
                    is_speech = False
                
                # State machine logic
                if state == TurnTakingState.WAITING_FOR_SPEECH:
                    if is_speech:
                        state = TurnTakingState.SPEECH_DETECTED
                        speech_duration_ms = frame_duration_ms
                        silence_duration_ms = 0
                        logger.info(
                            "Turn-taking: speech_start",
                            extra={
                                "event": "speech_start",
                                "state": state.name,
                                "timestamp_ms": time.time() * 1000,
                                "total_duration_ms": total_duration_ms,
                            }
                        )
                
                elif state == TurnTakingState.SPEECH_DETECTED:
                    if is_speech:
                        speech_duration_ms += frame_duration_ms
                        silence_duration_ms = 0
                    else:
                        state = TurnTakingState.SILENCE_AFTER_SPEECH
                        silence_duration_ms = frame_duration_ms
                        logger.info(
                            "Turn-taking: silence_detected",
                            extra={
                                "event": "silence_detected",
                                "state": state.name,
                                "speech_duration_ms": speech_duration_ms,
                                "total_duration_ms": total_duration_ms,
                            }
                        )
                
                elif state == TurnTakingState.SILENCE_AFTER_SPEECH:
                    if is_speech:
                        # Speech resumed, cancel silence detection
                        state = TurnTakingState.SPEECH_DETECTED
                        speech_duration_ms += frame_duration_ms
                        silence_duration_ms = 0
                        logger.info(
                            "Turn-taking: speech_resumed",
                            extra={
                                "event": "speech_resumed",
                                "state": state.name,
                                "total_duration_ms": total_duration_ms,
                            }
                        )
                    else:
                        silence_duration_ms += frame_duration_ms
                        
                        # Check if silence threshold met
                        if silence_duration_ms >= self.turn_taking_config.end_silence_duration_ms:
                            # Only enter grace period if minimum speech duration met
                            if speech_duration_ms >= self.turn_taking_config.min_speech_duration_ms:
                                state = TurnTakingState.GRACE_PERIOD
                                logger.info(
                                    "Turn-taking: silence_threshold_met",
                                    extra={
                                        "event": "silence_threshold_met",
                                        "state": state.name,
                                        "silence_duration_ms": silence_duration_ms,
                                        "speech_duration_ms": speech_duration_ms,
                                        "total_duration_ms": total_duration_ms,
                                    }
                                )
                
                elif state == TurnTakingState.GRACE_PERIOD:
                    if is_speech:
                        # Speech resumed during grace period, cancel finalization
                        state = TurnTakingState.SPEECH_DETECTED
                        speech_duration_ms += frame_duration_ms
                        silence_duration_ms = 0
                        logger.info(
                            "Turn-taking: grace_cancelled_speech_resumed",
                            extra={
                                "event": "grace_cancelled_speech_resumed",
                                "state": state.name,
                                "total_duration_ms": total_duration_ms,
                            }
                        )
                    else:
                        silence_duration_ms += frame_duration_ms
                        
                        # Check if grace period complete
                        grace_elapsed = silence_duration_ms - self.turn_taking_config.end_silence_duration_ms
                        if grace_elapsed >= self.turn_taking_config.post_speech_grace_ms:
                            state = TurnTakingState.FINALIZED
                            logger.info(
                                "Turn-taking: utterance_finalized",
                                extra={
                                    "event": "utterance_finalized",
                                    "state": state.name,
                                    "total_duration_ms": total_duration_ms,
                                    "speech_duration_ms": speech_duration_ms,
                                    "reason": "grace_complete",
                                }
                            )
                
                # Safety ceiling check
                if total_duration_ms >= (self.turn_taking_config.max_recording_duration_s * 1000):
                    logger.info(
                        "Turn-taking: max_duration_reached",
                        extra={
                            "event": "max_duration_reached",
                            "state": state.name,
                            "total_duration_ms": total_duration_ms,
                            "max_duration_ms": self.turn_taking_config.max_recording_duration_s * 1000,
                        }
                    )
                    if speech_duration_ms >= self.turn_taking_config.min_speech_duration_ms:
                        state = TurnTakingState.FINALIZED
                        logger.info(
                            "Turn-taking: utterance_finalized",
                            extra={
                                "event": "utterance_finalized",
                                "state": state.name,
                                "total_duration_ms": total_duration_ms,
                                "speech_duration_ms": speech_duration_ms,
                                "reason": "max_duration",
                            }
                        )
                    else:
                        # Max duration reached but insufficient speech, finalize anyway
                        state = TurnTakingState.FINALIZED
                        logger.warning(
                            "Turn-taking: finalized with insufficient speech",
                            extra={
                                "event": "utterance_finalized",
                                "state": state.name,
                                "total_duration_ms": total_duration_ms,
                                "speech_duration_ms": speech_duration_ms,
                                "reason": "max_duration_insufficient_speech",
                            }
                        )
            
            # Stop and close stream
            stream.stop()
            stream.close()
            
            # Concatenate all buffered frames
            audio_data = np.concatenate(audio_buffer)
            
            logger.info(
                f"VAD-based capture completed (samples={len(audio_data)}, "
                f"duration={total_duration_ms / 1000:.2f}s, speech={speech_duration_ms / 1000:.2f}s)"
            )
            
            return AudioRecording(
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
            )
        
        except Exception as e:
            logger.error(f"VAD-based capture failed: {e}")
            # Fallback to fixed-duration capture
            logger.warning("Falling back to fixed-duration capture")
            return self.capture(duration=self.default_duration)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        logger.debug("AudioCapture context exited")