"""
Text-to-Speech module for Sovereign.

Speaks responses back to user using Windows native TTS (pyttsx3).
Interface designed to support future TTS providers like Piper.
"""

import asyncio
import logging
from typing import Optional

import pyttsx3

from ..config import TTSConfig

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-Speech engine for speaking responses.
    
    Primary implementation uses Windows native TTS via pyttsx3.
    Engine is loaded lazily on first speak to reduce startup time.
    Supports voice selection, speech rate, and volume control.
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize the text-to-speech engine.
        
        Args:
            config: TTS configuration with provider, voice, and rate settings
        """
        self.config = config
        self._engine: Optional[pyttsx3.Engine] = None
        
        logger.info(
            f"TextToSpeech initialized (provider={config.provider}, "
            f"voice={config.voice or 'default'}, rate={config.rate})"
        )
    
    def _load_engine(self) -> None:
        """
        Lazily load the TTS engine on first speak.
        
        Initializes pyttsx3 engine, applies voice and rate settings from config,
        and handles missing voices gracefully with fallback to default.
        
        Raises:
            RuntimeError: If engine initialization fails
        """
        if self._engine is not None:
            return
        
        try:
            logger.info("Loading TTS engine (first speak)")
            
            # Initialize pyttsx3 engine
            # NOTE: For future Piper TTS integration, replace this section with
            # Piper-specific initialization while maintaining the same interface
            self._engine = pyttsx3.init()
            
            # Apply voice selection if specified
            if self.config.voice:
                voices = self._engine.getProperty('voices')
                voice_found = False
                
                for voice in voices:
                    if self.config.voice.lower() in voice.name.lower():
                        self._engine.setProperty('voice', voice.id)
                        voice_found = True
                        logger.info(f"Voice set to: {voice.name}")
                        break
                
                if not voice_found:
                    logger.warning(
                        f"Voice '{self.config.voice}' not found - using default voice"
                    )
                    available_voices = [v.name for v in voices]
                    logger.debug(f"Available voices: {available_voices}")
            
            # Apply speech rate
            # pyttsx3 rate is in words per minute (default ~200)
            # config.rate is a multiplier (0.5-2.0), so convert to wpm
            default_rate = self._engine.getProperty('rate')
            new_rate = int(default_rate * self.config.rate)
            self._engine.setProperty('rate', new_rate)
            logger.info(f"Speech rate set to {new_rate} wpm (multiplier: {self.config.rate})")
            
            # Set volume to max (0.0-1.0 range)
            # NOTE: Could be made configurable via TTSConfig if needed
            self._engine.setProperty('volume', 1.0)
            
            logger.info("TTS engine loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load TTS engine: {e}")
            raise RuntimeError(f"TTS engine initialization failed: {e}")
    
    def speak(self, text: str) -> None:
        """
        Speak text synchronously (blocking).
        
        Converts text to speech using the configured TTS engine and voice.
        This method blocks until speech completes.
        
        Args:
            text: The text to speak
        
        Raises:
            RuntimeError: If speech synthesis fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for speech")
            return
        
        # Ensure engine is loaded
        self._load_engine()
        
        try:
            logger.info(f"Speaking text (length={len(text)}): {text[:50]}...")
            
            # Queue text for speech
            self._engine.say(text)
            
            # Block until speech completes
            # NOTE: For Piper TTS, this would be replaced with Piper's
            # synthesis and audio playback logic
            self._engine.runAndWait()
            
            # CRITICAL FIX: On Windows, pyttsx3/SAPI5 leaves the engine in a bad state
            # after the first runAndWait() call, causing subsequent calls to silently fail.
            # Stopping and reinitializing the engine after each speak ensures reliable
            # multi-turn voice output. This only affects the TTS engine state, NOT
            # Sovereign's main loop or conversation context.
            self._engine.stop()
            self._engine = None  # Force reload on next speak
            
            logger.info("Speech completed")
        
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise RuntimeError(f"Failed to speak text: {e}")
    
    async def speak_async(self, text: str) -> None:
        """
        Speak text asynchronously (non-blocking).
        
        Converts text to speech in a background thread to avoid blocking
        the event loop. Useful for async/await workflows.
        
        Args:
            text: The text to speak
        
        Raises:
            RuntimeError: If speech synthesis fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for async speech")
            return
        
        try:
            logger.info(f"Speaking text asynchronously (length={len(text)})")
            
            # Run blocking speak() in executor to avoid blocking event loop
            # NOTE: For Piper TTS, this could use native async I/O if available
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.speak, text)
            
            logger.info("Async speech completed")
        
        except Exception as e:
            logger.error(f"Async speech synthesis failed: {e}")
            raise RuntimeError(f"Failed to speak text asynchronously: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.
        
        Stops any ongoing speech and releases TTS engine resources.
        """
        if self._engine is not None:
            try:
                self._engine.stop()
                logger.debug("TTS engine stopped")
            except Exception as e:
                logger.warning(f"Error stopping TTS engine: {e}")
        
        logger.debug("TextToSpeech context exited")