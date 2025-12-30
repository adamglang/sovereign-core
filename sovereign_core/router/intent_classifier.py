"""Intent classification logic for the Router."""

import logging
from typing import List

from ..brain.llm_interface import LLMProvider
from .models import IntentType
from .prompts import get_classification_prompt

logger = logging.getLogger(__name__)


def classify_intent(
    utterance: str,
    llm_provider: LLMProvider,
    action_keywords: List[str],
) -> IntentType:
    """
    Classify a user utterance into CONVERSATIONAL, ACTION, or CLARIFICATION_NEEDED.
    
    Uses an LLM to analyze the user's intent based on their utterance.
    This is intentionally LLM-based rather than rule-based to handle natural
    language variations and nuanced requests.
    
    Args:
        utterance: The user's spoken or typed input
        llm_provider: LLM provider instance for classification
        action_keywords: List of keywords that indicate action requests
        
    Returns:
        IntentType enum value (CONVERSATIONAL, ACTION, or CLARIFICATION_NEEDED)
        
    Raises:
        Exception: If LLM classification fails or returns invalid classification
    """
    logger.debug(f"Classifying intent for utterance: {utterance}")
    
    system_prompt = get_classification_prompt(action_keywords)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User utterance: {utterance}"},
    ]
    
    try:
        # Use low temperature for consistent classification
        response = llm_provider.generate_response(messages, temperature=0.1)
        classification = response.strip().upper()
        
        logger.debug(f"LLM classification response: {classification}")
        
        # Parse the classification response
        if "CONVERSATIONAL" in classification:
            logger.info(f"Classified as CONVERSATIONAL: {utterance}")
            return IntentType.CONVERSATIONAL
        elif "ACTION" in classification:
            logger.info(f"Classified as ACTION: {utterance}")
            return IntentType.ACTION
        elif "CLARIFICATION" in classification:
            logger.info(f"Classified as CLARIFICATION_NEEDED: {utterance}")
            return IntentType.CLARIFICATION_NEEDED
        else:
            # Default to conversational if unclear
            logger.warning(
                f"Unclear classification '{classification}' for utterance: {utterance}. "
                f"Defaulting to CONVERSATIONAL."
            )
            return IntentType.CONVERSATIONAL
            
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        # Safe fallback: treat as conversational to avoid unintended actions
        logger.warning("Defaulting to CONVERSATIONAL due to classification failure")
        return IntentType.CONVERSATIONAL