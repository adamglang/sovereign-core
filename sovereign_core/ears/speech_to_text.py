"""
Speech-to-Text module for Sovereign using faster-whisper.

Transcribes captured audio to text with GPU acceleration support.
"""

import logging
from typing import Optional

import numpy as np
import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel

from ..config import STTConfig
from .audio_capture import AudioRecording

logger = logging.getLogger(__name__)


class TranscriptionResult(BaseModel):
    """Result of speech-to-text transcription."""
    
    text: str
    confidence: float
    language: Optional[str] = None


class SpeechToText:
    """
    Transcribes audio to text using faster-whisper.
    
    Supports GPU acceleration (CUDA) with automatic fallback to CPU.
    Model is loaded lazily on first transcription to reduce startup time.
    """
    
    def __init__(self, config: STTConfig):
        """
        Initialize the speech-to-text engine.
        
        Args:
            config: STT configuration with model size, device, and compute type
        """
        self.config = config
        self._model: Optional[WhisperModel] = None
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        
        logger.info(
            f"SpeechToText initialized (model_size={config.model_size}, "
            f"device={config.device}, compute_type={config.compute_type})"
        )
    
    def _load_model(self) -> None:
        """
        Lazily load the Whisper model on first transcription.
        
        Auto-detects CUDA availability and falls back to CPU if needed.
        Updates device and compute_type based on availability.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model is not None:
            return
        
        try:
            logger.info("Loading Whisper model (first transcription)")
            
            # Auto-detect device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._device = "cuda"
                self._compute_type = self.config.compute_type
                logger.info(
                    f"CUDA available - using GPU acceleration "
                    f"(compute_type={self._compute_type})"
                )
            else:
                self._device = "cpu"
                self._compute_type = "int8"
                if self.config.device == "cuda":
                    logger.warning(
                        "CUDA requested but not available - falling back to CPU"
                    )
                logger.info(f"Using CPU (compute_type={self._compute_type})")
            
            # Load model
            self._model = WhisperModel(
                self.config.model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self.config.model_dir,
            )
            
            logger.info(
                f"Whisper model loaded successfully "
                f"(model_size={self.config.model_size}, device={self._device})"
            )
        
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failed: {e}")
    
    def transcribe(self, recording: AudioRecording) -> TranscriptionResult:
        """
        Transcribe audio recording to text.
        
        Converts numpy audio data to float32 format expected by faster-whisper,
        performs transcription, and returns text with confidence metadata.
        
        Args:
            recording: Audio recording with data and sample rate
        
        Returns:
            TranscriptionResult: Transcribed text, confidence score, and detected language
        
        Raises:
            RuntimeError: If transcription fails
            ValueError: If audio data is empty or invalid
        """
        # Validate input
        if recording.audio_data is None or len(recording.audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=None,
            )
        
        # Ensure model is loaded
        self._load_model()
        
        try:
            logger.info(
                f"Starting transcription (samples={len(recording.audio_data)}, "
                f"sample_rate={recording.sample_rate})"
            )
            
            # Convert int16 audio to float32 normalized to [-1.0, 1.0]
            audio_float32 = recording.audio_data.astype(np.float32) / 32768.0
            
            # Transcribe with faster-whisper
            segments, info = self._model.transcribe(
                audio_float32,
                beam_size=5,
                language=None,  # Auto-detect language
            )
            
            # Collect segments and compute average confidence
            text_parts = []
            confidences = []
            
            for segment in segments:
                text_parts.append(segment.text)
                confidences.append(segment.avg_logprob)
            
            # Handle case where no speech was detected
            if not text_parts:
                logger.info("No speech detected in audio")
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=info.language if info else None,
                )
            
            # Combine segments
            full_text = " ".join(text_parts).strip()
            
            # Convert log probability to confidence score (0.0 to 1.0)
            # avg_logprob ranges from -inf to 0, we normalize to 0-1 range
            avg_confidence = np.mean(confidences) if confidences else 0.0
            confidence_score = np.exp(avg_confidence)  # Convert log prob to probability
            
            logger.info(
                f"Transcription completed (text_length={len(full_text)}, "
                f"confidence={confidence_score:.2f}, language={info.language})"
            )
            
            return TranscriptionResult(
                text=full_text,
                confidence=float(confidence_score),
                language=info.language if info else None,
            )
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._model is not None:
            logger.debug("SpeechToText context exited")