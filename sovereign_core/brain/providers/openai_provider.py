"""
OpenAI LLM provider implementation.

This module implements the LLMProvider interface for OpenAI's GPT models,
supporting both conversational and structured JSON outputs.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

from openai import OpenAI
from openai.types.chat import ChatCompletion

from ..llm_interface import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation of the LLM provider interface.
    
    Supports GPT-4, GPT-4 Turbo, GPT-3.5, and other OpenAI chat models.
    Implements retry logic with exponential backoff for reliability.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: OpenAI model name (e.g., 'gpt-4o-mini', 'gpt-4-turbo')
            temperature: Default temperature for responses
            max_tokens: Maximum tokens in response
            max_retries: Number of retry attempts for failed requests
            retry_delay: Initial delay in seconds between retries (exponential backoff)
        
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set. "
                "Get your API key from https://platform.openai.com/api-keys"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a conversational response using OpenAI's chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        
        Returns:
            str: The assistant's response text
            
        Raises:
            Exception: On API errors after all retries exhausted
        """
        logger.debug(f"Generating response with {len(messages)} messages")
        
        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                
                result = response.choices[0].message.content or ""
                logger.debug(f"Generated response: {result[:100]}...")
                return result
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted: {str(e)}")
                    raise
        
        return ""
    
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using OpenAI's structured output feature.
        
        Uses the response_format parameter to enforce JSON schema compliance.
        For OpenAI, this uses the 'json_object' mode which instructs the model
        to output valid JSON.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: JSON schema specification (currently uses type: json_object)
        
        Returns:
            dict: Parsed JSON response
            
        Raises:
            Exception: On API errors or JSON parsing failures
        """
        logger.debug(f"Generating structured output with {len(messages)} messages")
        
        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.default_temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)
                
                logger.debug(f"Generated structured output: {result}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw content: {content}")
                raise
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted: {str(e)}")
                    raise
        
        return {}
    
    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming responses.
        
        Returns:
            bool: True (OpenAI supports streaming)
        """
        return True