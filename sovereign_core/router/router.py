"""Main Router component for sovereign-core."""

import logging
import time
from typing import Any, Dict, List

from ..brain.llm_interface import LLMProvider
from ..ipc.database import create_command
from .intent_classifier import classify_intent
from .models import IntentType, RouterResponse
from .prompts import (
    get_action_extraction_prompt,
    get_clarification_prompt,
    get_conversational_prompt,
)

logger = logging.getLogger(__name__)


class Router:
    """
    Router component that enforces the safety boundary between conversation and action modes.
    
    This is the MOST CRITICAL component in sovereign-core. It analyzes user utterances
    and decides whether to respond conversationally or queue an action for execution.
    
    SAFETY is achieved through SEPARATION:
    - Conversational mode: Natural language output ONLY, no actions
    - Action mode: Structured JSON output ONLY, written to IPC
    - Ambiguous requests: Ask for clarification instead of guessing
    
    The Router NEVER executes actions directly. It only writes commands to the IPC
    database for sovereign-executor to process.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        db_path: str,
        action_keywords: List[str],
        context_messages: int = 10,
    ):
        """
        Initialize the Router.
        
        Args:
            llm_provider: LLM provider instance for all routing tasks
            db_path: Path to IPC database for writing action commands
            action_keywords: List of keywords that indicate action requests
            context_messages: Number of conversation messages to pass as context
        """
        self.llm_provider = llm_provider
        self.db_path = db_path
        self.action_keywords = action_keywords
        self.context_messages = context_messages
        
        logger.info("Router initialized")
    
    def route(
        self,
        utterance: str,
        conversation_history: List[dict] | None = None,
    ) -> RouterResponse:
        """
        Route a user utterance to the appropriate response type.
        
        This is the main entry point for the Router. It orchestrates the full
        routing pipeline:
        1. Classify the intent (CONVERSATIONAL, ACTION, or CLARIFICATION_NEEDED)
        2. Generate appropriate response based on classification
        3. For actions, validate and write to IPC database
        
        Args:
            utterance: The user's spoken or typed input
            conversation_history: Optional list of previous conversation messages
                                 for context in conversational responses
                                 
        Returns:
            RouterResponse with one of three types:
            - CONVERSATIONAL: Natural language response
            - ACTION: Structured intent written to IPC, includes command_id
            - CLARIFICATION_NEEDED: Question to clarify ambiguous request
            
        Raises:
            Exception: On critical failures in routing logic
        """
        if conversation_history is None:
            conversation_history = []
        
        logger.info(f"Routing utterance: {utterance}")
        
        # Step 1: Classify the intent
        classify_start = time.perf_counter()
        intent_type = classify_intent(
            utterance=utterance,
            llm_provider=self.llm_provider,
            action_keywords=self.action_keywords,
        )
        classify_duration = time.perf_counter() - classify_start
        logger.info(f"Intent classification took {classify_duration:.2f}s")
        
        # Step 2: Handle based on classification
        if intent_type == IntentType.CONVERSATIONAL:
            handle_start = time.perf_counter()
            result = self._handle_conversational(utterance, conversation_history)
            handle_duration = time.perf_counter() - handle_start
            logger.info(f"Conversational handling took {handle_duration:.2f}s")
            return result
        
        elif intent_type == IntentType.ACTION:
            return self._handle_action(utterance)
        
        elif intent_type == IntentType.CLARIFICATION_NEEDED:
            return self._handle_clarification(utterance)
        
        else:
            # Should never happen, but safe fallback
            logger.error(f"Unknown intent type: {intent_type}")
            return RouterResponse(
                type=IntentType.CONVERSATIONAL,
                conversational_response="I'm not sure I understand. Could you rephrase that?",
            )
    
    def _handle_conversational(
        self,
        utterance: str,
        conversation_history: List[dict],
    ) -> RouterResponse:
        """
        Generate a conversational response.
        
        Args:
            utterance: The user's question or statement
            conversation_history: Previous conversation for context
            
        Returns:
            RouterResponse with conversational_response
        """
        logger.debug("Handling conversational intent")
        
        system_prompt = get_conversational_prompt(conversation_history, self.context_messages)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ]
        
        try:
            llm_start = time.perf_counter()
            response = self.llm_provider.generate_response(
                messages=messages,
                temperature=0.7,  # More creative for conversation
            )
            llm_duration = time.perf_counter() - llm_start
            
            logger.info(f"LLM response generation took {llm_duration:.2f}s")
            logger.info(f"Generated conversational response: {response[:100]}...")
            
            return RouterResponse(
                type=IntentType.CONVERSATIONAL,
                conversational_response=response,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {str(e)}")
            # Safe fallback response
            return RouterResponse(
                type=IntentType.CONVERSATIONAL,
                conversational_response=(
                    "I'm having trouble processing that right now. "
                    "Could you try asking in a different way?"
                ),
            )
    
    def _handle_action(self, utterance: str) -> RouterResponse:
        """
        Extract action intent and write to IPC database.
        
        CRITICAL: This method NEVER executes actions. It only writes structured
        action intents to the IPC database for sovereign-executor to process.
        
        Args:
            utterance: The user's action command
            
        Returns:
            RouterResponse with action_intent and command_id
        """
        logger.debug("Handling action intent")
        
        system_prompt = get_action_extraction_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User command: {utterance}"},
        ]
        
        try:
            # Extract structured action intent
            action_data = self.llm_provider.generate_structured(
                messages=messages,
                response_format={"type": "json_object"},
            )
            
            logger.debug(f"Extracted action data: {action_data}")
            
            # Validate action format
            if "action" not in action_data or "params" not in action_data:
                logger.error(f"Invalid action format: {action_data}")
                # Fall back to clarification
                return RouterResponse(
                    type=IntentType.CLARIFICATION_NEEDED,
                    clarification_question=(
                        "I understood you want me to do something, but I'm not sure "
                        "exactly what. Could you be more specific?"
                    ),
                )
            
            action = action_data["action"]
            params = action_data["params"]
            
            # Validate action is supported (basic sanity check)
            if not action.startswith("spotify."):
                logger.warning(f"Unsupported action category: {action}")
                return RouterResponse(
                    type=IntentType.CLARIFICATION_NEEDED,
                    clarification_question=(
                        f"I don't know how to perform '{action}'. "
                        "I can help with music playback. What would you like me to play?"
                    ),
                )
            
            # Write to IPC database - this is the ONLY action execution point
            command_id = create_command(
                db_path=self.db_path,
                action=action,
                params=params,
                speaker_id=None,  # Future enhancement
            )
            
            logger.info(
                f"Created command {command_id} for action '{action}' with params {params}"
            )
            
            return RouterResponse(
                type=IntentType.ACTION,
                action_intent=action_data,
                command_id=command_id,
            )
            
        except Exception as e:
            logger.error(f"Failed to handle action intent: {str(e)}")
            # Safe fallback to clarification
            return RouterResponse(
                type=IntentType.CLARIFICATION_NEEDED,
                clarification_question=(
                    "I'm having trouble understanding that command. "
                    "Could you try phrasing it differently?"
                ),
            )
    
    def _handle_clarification(self, utterance: str) -> RouterResponse:
        """
        Generate a clarification question for ambiguous requests.
        
        Args:
            utterance: The ambiguous user input
            
        Returns:
            RouterResponse with clarification_question
        """
        logger.debug("Handling clarification needed")
        
        system_prompt = get_clarification_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User's ambiguous request: {utterance}"},
        ]
        
        try:
            question = self.llm_provider.generate_response(
                messages=messages,
                temperature=0.7,
            )
            
            logger.info(f"Generated clarification question: {question}")
            
            return RouterResponse(
                type=IntentType.CLARIFICATION_NEEDED,
                clarification_question=question,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate clarification: {str(e)}")
            # Safe fallback question
            return RouterResponse(
                type=IntentType.CLARIFICATION_NEEDED,
                clarification_question=(
                    "I'm not sure I understand what you'd like me to do. "
                    "Could you provide more details?"
                ),
            )