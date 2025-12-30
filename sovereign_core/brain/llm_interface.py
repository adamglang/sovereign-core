"""
Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
enabling zero-friction model swapping through configuration changes only.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class and implement
    the required methods. This ensures consistent behavior across different
    LLM providers (OpenAI, Anthropic, Gemini, local models, etc.).
    """
    
    @abstractmethod
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7
    ) -> str:
        """
        Generate a conversational response from the LLM.
        
        Used for natural language interactions where the response is
        a human-readable string rather than structured data.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Role can be 'system', 'user', or 'assistant'.
                     Example: [{"role": "user", "content": "Hello!"}]
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        
        Returns:
            str: The LLM's response as a string
            
        Raises:
            Exception: Provider-specific errors (network, API, rate limit, etc.)
        """
        pass
    
    @abstractmethod
    def generate_structured(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from the LLM.
        
        Used for action intent parsing where the response must match a
        specific JSON schema. The LLM is prompted to return valid JSON
        that conforms to the provided format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: JSON schema or format specification that the
                           response must conform to
        
        Returns:
            dict: Parsed JSON response matching the requested format
            
        Raises:
            Exception: Provider-specific errors or JSON parsing failures
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming responses.
        
        Returns:
            bool: True if streaming is supported, False otherwise
        """
        pass