"""
Text-to-Speech module for Sovereign.

Uses Piper TTS for high-quality neural speech synthesis with GPU acceleration.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig

from ..config import TTSConfig

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-Speech engine using Piper for neural voice synthesis.
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize the text-to-speech engine.
        
        Args:
            config: TTS configuration with voice model and GPU settings
        """
        self.config = config
        self.voice: Optional[PiperVoice] = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(
            f"TextToSpeech initialized (voice_model={config.voice_model}, "
            f"speaker_id={config.speaker_id}, use_cuda={config.use_cuda})"
        )
    
    def _load_voice(self) -> None:
        """
        Lazily load the Piper voice model on first speak.
        
        Raises:
            FileNotFoundError: If model files are missing
            RuntimeError: If voice loading fails
        """
        if self.voice is not None:
            return
        
        logger.info("Loading Piper voice model (first speak)")
        
        model_path = Path(f"./models/piper/{self.config.voice_model}.onnx")
        config_path = Path(f"./models/piper/{self.config.voice_model}.onnx.json")
        
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Piper model files not found: {model_path}\n"
                f"Please run setup to download models. See SETUP.md for details."
            )
        
        try:
            self.voice = PiperVoice.load(
                str(model_path),
                use_cuda=self.config.use_cuda
            )
            
            device = "GPU (CUDA)" if self.config.use_cuda else "CPU"
            logger.info(f"Piper voice loaded successfully on {device}")
        
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            raise RuntimeError(f"Voice model loading failed: {e}")
    
    def speak(self, text: str) -> None:
        """
        Speak text synchronously (blocking).
        
        Args:
            text: The text to speak
        
        Raises:
            ValueError: If text is empty
            RuntimeError: If synthesis or playback fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for speech")
            return
        
        self._load_voice()
        
        try:
            logger.info(f"Speaking text (length={len(text)}): {text[:50]}...")
            
            # Create synthesis config with speaker_id if specified
            syn_config = None
            if self.config.speaker_id is not None:
                syn_config = SynthesisConfig(speaker_id=self.config.speaker_id)
            
            # Synthesize returns Iterable[AudioChunk], extract audio arrays
            audio_chunks = []
            sample_rate = None
            for chunk in self.voice.synthesize(text, syn_config):
                audio_chunks.append(chunk.audio_int16_array)
                if sample_rate is None:
                    sample_rate = chunk.sample_rate
            
            # Concatenate all audio chunks into single array
            audio_array = np.concatenate(audio_chunks)
            
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
            
            logger.info("Speech completed")
        
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise RuntimeError(f"Failed to speak text: {e}")
    
    async def speak_async(self, text: str) -> None:
        """
        Speak text asynchronously (non-blocking).
        
        Args:
            text: The text to speak
        
        Raises:
            ValueError: If text is empty
            RuntimeError: If synthesis or playback fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for async speech")
            return
        
        try:
            logger.info(f"Speaking text asynchronously (length={len(text)})")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.speak, text)
            
            logger.info("Async speech completed")
        
        except Exception as e:
            logger.error(f"Async speech synthesis failed: {e}")
            raise RuntimeError(f"Failed to speak text asynchronously: {e}")
    
    def cleanup(self) -> None:
        """
        Release TTS resources.
        """
        self.executor.shutdown(wait=True)
        self.voice = None
        logger.debug("TTS resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.
        """
        self.cleanup()
        logger.debug("TextToSpeech context exited")
