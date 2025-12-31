"""
Main orchestrator for Sovereign Core voice assistant.

This module contains the SovereignCore class that coordinates all components
to create a complete local-first voice assistant with the following flow:

1. Wake word detection ("Hey Sovereign")
2. Audio capture (user speaks)
3. Speech-to-text transcription
4. Intent routing (conversation vs action)
5. Response generation (LLM or IPC command)
6. Text-to-speech output
7. Loop back to listening

The orchestrator handles initialization, lifecycle management, error handling,
and graceful shutdown.
"""

import logging
from logging.handlers import RotatingFileHandler
import signal
import sys
from pathlib import Path
from typing import Optional

from .brain.llm_factory import get_llm_provider
from .config import load_config
from .ears.audio_capture import AudioCapture
from .ears.speech_to_text import SpeechToText
from .ears.wake_word_detector import WakeWordDetector
from .ipc.database import init_database
from .mouth.text_to_speech import TextToSpeech
from .router.models import IntentType
from .router.router import Router

logger = logging.getLogger(__name__)


class SovereignCore:
    """
    Main orchestrator for the Sovereign voice assistant.
    
    Coordinates all components in a cohesive loop:
    - Wake word detection
    - Audio capture
    - Speech-to-text
    - Intent routing
    - Response generation
    - Text-to-speech
    
    Handles component lifecycle, error recovery, and graceful shutdown.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Sovereign Core orchestrator.
        
        Args:
            config_path: Optional path to config.yaml. If None, uses default.
        """
        logger.info("Initializing Sovereign Core...")
        
        # Load configuration
        self.config = load_config(config_path)
        self._setup_logging()
        
        # Component instances
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.router: Optional[Router] = None
        
        # State
        self.running = False
        self.conversation_history = []
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("Sovereign Core initialized successfully")
    
    def _setup_logging(self) -> None:
        """Configure logging based on config settings."""
        log_config = self.config.logging
        
        # Create logs directory if needed
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_config.level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_config.file,
            maxBytes=log_config.max_bytes,
            backupCount=log_config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_config.level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        if log_config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_config.level)
            console_formatter = logging.Formatter(
                "%(levelname)s: %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Suppress noisy third-party libraries to prevent log bloat
        third_party_level = getattr(logging, log_config.third_party_level)
        noisy_loggers = [
            'pvporcupine',
            'pyaudio',
            'faster_whisper',
            'openai',
            'httpx',
            'httpcore',
            'urllib3',
            'sounddevice',
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(third_party_level)
        
        logger.info(
            f"Logging configured: level={log_config.level}, file={log_config.file}, "
            f"max_size={log_config.max_bytes / 1_048_576:.1f}MB, "
            f"backups={log_config.backup_count}, third_party_level={log_config.third_party_level}"
        )
    
    def _initialize_components(self) -> None:
        """Initialize all voice assistant components in correct order."""
        logger.info("Initializing components...")
        
        # Initialize IPC database first
        logger.info(f"Initializing IPC database at {self.config.ipc.database_path}")
        init_database(self.config.ipc.database_path)
        
        # Initialize LLM provider
        logger.info(f"Initializing LLM provider: {self.config.llm.provider}")
        llm_provider = get_llm_provider(
            provider_name=self.config.llm.provider,
            config={
                "model": self.config.llm.model,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
            },
        )
        
        # Initialize Router
        logger.info("Initializing Router")
        self.router = Router(
            llm_provider=llm_provider,
            db_path=self.config.ipc.database_path,
            action_keywords=self.config.router.action_keywords,
            context_messages=self.config.conversation.context_messages,
        )
        
        # Initialize wake word detector
        logger.info("Initializing wake word detector")
        self.wake_word_detector = WakeWordDetector(
            config=self.config.wake_word,
        )
        
        # Initialize audio capture
        logger.info("Initializing audio capture")
        self.audio_capture = AudioCapture(config=self.config.audio)
        
        # Initialize speech-to-text
        logger.info(f"Initializing STT with model: {self.config.stt.model_size}")
        self.stt = SpeechToText(config=self.config.stt)
        
        # Initialize text-to-speech
        logger.info(f"Initializing TTS with provider: {self.config.tts.provider}")
        self.tts = TextToSpeech(config=self.config.tts)
        
        logger.info("All components initialized")
    
    def _handle_response(self, response_text: str) -> None:
        """
        Handle a response by speaking it and updating conversation history.
        
        Args:
            response_text: The text to speak
        """
        logger.info(f"Assistant response: {response_text}")
        
        # Speak the response
        logger.debug("Converting response to speech...")
        self.tts.speak(response_text)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })
        
        # Keep history manageable
        max_messages = self.config.conversation.max_history_messages
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
    
    def main_loop(self) -> None:
        """
        Main processing loop for the voice assistant.
        
        Flow:
        1. Wait for wake word
        2. Capture user audio
        3. Transcribe to text
        4. Route to appropriate handler
        5. Generate response
        6. Speak response
        7. Repeat
        
        This loop runs continuously until stop() is called or an
        unrecoverable error occurs.
        """
        logger.info("Entering main loop")
        
        # Log once before entering the loop
        logger.info("Listening for wake word...")
        print("\n[*] Listening for 'Hey Sovereign'...")
        
        while self.running:
            try:
                # Step 1: Wait for wake word (blocks until detected)
                wake_detected = self.wake_word_detector.wait_for_wake_word()
                
                if not wake_detected:
                    if not self.running:
                        break
                    # If wake_detected is False but we're still running, something is wrong
                    logger.error("Wake word detector returned False - this should block!")
                    logger.error("This usually means the wake word detector failed to start properly")
                    print("[!] Wake word detector error - check logs for details")
                    self.tts.speak("The wake word detector encountered an error.")
                    break
                
                if not self.running:
                    break
                
                logger.info("Wake word detected!")
                print("[+] Wake word detected! Speak now...")
                
                # Step 2: Capture user audio
                logger.info("Capturing audio...")
                audio_recording = self.audio_capture.capture(duration=5)
                
                if not audio_recording or not audio_recording.audio_data.size:
                    logger.warning("No audio captured")
                    self.tts.speak("I didn't hear anything. Please try again.")
                    continue
                
                logger.debug(f"Captured {len(audio_recording.audio_data)} audio samples")
                
                # Step 3: Transcribe audio to text
                logger.info("Transcribing audio...")
                print("[~] Transcribing...")
                
                transcription_result = self.stt.transcribe(audio_recording)
                utterance = transcription_result.text
                
                if not utterance or utterance.strip() == "":
                    logger.warning("Transcription resulted in empty text")
                    self.tts.speak("I couldn't understand that. Please try again.")
                    continue
                
                logger.info(f"User said: {utterance}")
                print(f"[>] You: {utterance}")
                
                # Add user utterance to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": utterance,
                })
                
                # Step 4: Route to appropriate handler
                logger.info("Routing request...")
                print("[~] Processing...")
                
                router_response = self.router.route(
                    utterance=utterance,
                    conversation_history=self.conversation_history,
                )
                
                # Step 5 & 6: Generate and speak response based on intent type
                if router_response.type == IntentType.CONVERSATIONAL:
                    # Conversational response
                    response_text = router_response.conversational_response
                    self._handle_response(response_text)
                
                elif router_response.type == IntentType.ACTION:
                    # Action was queued to IPC
                    action_intent = router_response.action_intent
                    command_id = router_response.command_id
                    
                    logger.info(
                        f"Action queued: {action_intent['action']} "
                        f"with params {action_intent['params']} (command_id={command_id})"
                    )
                    
                    # Acknowledge the action
                    response_text = (
                        f"I've queued that action. "
                        f"Command {command_id} will be processed by the executor."
                    )
                    self._handle_response(response_text)
                
                elif router_response.type == IntentType.CLARIFICATION_NEEDED:
                    # Ask for clarification
                    response_text = router_response.clarification_question
                    self._handle_response(response_text)
                
                else:
                    # Should never happen, but handle gracefully
                    logger.error(f"Unknown response type: {router_response.type}")
                    response_text = "I encountered an error processing that request."
                    self._handle_response(response_text)
                
                # After processing, resume listening
                logger.info("Listening for wake word...")
                print("\n[*] Listening for 'Hey Sovereign'...")
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                logger.info("Keyboard interrupt received")
                break
            
            except Exception as e:
                # Log error but keep running
                logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                print(f"[!] Error: {str(e)}")
                
                # Try to inform user and continue
                try:
                    self.tts.speak(
                        "I encountered an error. Please try your request again."
                    )
                except Exception as tts_error:
                    logger.error(f"Failed to speak error message: {str(tts_error)}")
                
                # Resume listening after error
                logger.info("Listening for wake word...")
                print("\n[*] Listening for 'Hey Sovereign'...")
        
        logger.info("Exited main loop")
    
    def start(self) -> None:
        """
        Start the voice assistant.
        
        This sets up signal handlers, starts the main loop, and handles
        cleanup on exit.
        """
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("Starting Sovereign Core voice assistant")
        print("\n" + "=" * 60)
        print("*** Sovereign Core Voice Assistant ***")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            self.main_loop()
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Stop the voice assistant and clean up resources.
        
        This method is idempotent and safe to call multiple times.
        """
        if not self.running:
            return
        
        logger.info("Stopping Sovereign Core...")
        print("\n[*] Shutting down gracefully...")
        
        self.running = False
        
        # Clean up components in reverse order of initialization
        if self.wake_word_detector:
            try:
                logger.debug("Cleaning up wake word detector")
                self.wake_word_detector.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up wake word detector: {e}")
        
        if self.audio_capture:
            try:
                logger.debug("Cleaning up audio capture")
                self.audio_capture.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio capture: {e}")
        
        if self.stt:
            try:
                logger.debug("Cleaning up STT")
                self.stt.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up STT: {e}")
        
        logger.info("Sovereign Core stopped")
        print("[+] Shutdown complete")


def main():
    """CLI entry point for Sovereign Core."""
    try:
        # Create and start the voice assistant
        sovereign = SovereignCore()
        sovereign.start()
        
    except FileNotFoundError as e:
        print(f"\n[!] Configuration Error: {str(e)}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n[!] Configuration Error: {str(e)}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n[!] Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()