"""Pydantic models for the Router component."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Intent classification types."""
    
    CONVERSATIONAL = "conversational"
    ACTION = "action"
    CLARIFICATION_NEEDED = "clarification_needed"


class RouterResponse(BaseModel):
    """
    Response from the Router after processing a user utterance.
    
    This is a discriminated union based on the intent type. Only one payload
    field will be populated based on the type:
    - CONVERSATIONAL: conversational_response contains natural language
    - ACTION: action_intent contains action dict, command_id is set
    - CLARIFICATION_NEEDED: clarification_question contains the question
    """
    
    type: IntentType
    conversational_response: Optional[str] = None
    action_intent: Optional[dict[str, Any]] = None
    clarification_question: Optional[str] = None
    command_id: Optional[int] = None
    
    def model_post_init(self, __context: Any) -> None:
        """
        Validate that exactly one payload field is set based on type.
        
        This enforces the discriminated union at runtime.
        """
        if self.type == IntentType.CONVERSATIONAL:
            if not self.conversational_response:
                raise ValueError("CONVERSATIONAL type requires conversational_response")
        elif self.type == IntentType.ACTION:
            if not self.action_intent or not self.command_id:
                raise ValueError("ACTION type requires action_intent and command_id")
        elif self.type == IntentType.CLARIFICATION_NEEDED:
            if not self.clarification_question:
                raise ValueError("CLARIFICATION_NEEDED type requires clarification_question")