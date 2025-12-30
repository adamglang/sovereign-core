"""
Factory for creating LLM provider instances.

This module provides a factory function that instantiates the appropriate
LLM provider based on configuration, enabling zero-friction model swapping.
"""

import logging
from typing import Any, Dict

from .llm_interface import LLMProvider
from .providers import OpenAIProvider

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider_name: str = "openai",
    config: Dict[str, Any] | None = None,
) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider instance.
    
    This function instantiates the correct provider based on the provider name
    and configuration. Adding new providers requires only updating this factory
    and implementing the new provider class.
    
    Args:
        provider_name: Name of the provider ("openai", future: "anthropic", "gemini", etc.)
        config: Optional configuration dictionary with provider-specific settings.
                If None, uses default values.
    
    Returns:
        LLMProvider: Instance of the requested provider
        
    Raises:
        ValueError: If provider_name is not recognized
        
    Example:
        >>> config = {"model": "gpt-4o-mini", "temperature": 0.7}
        >>> provider = get_llm_provider("openai", config)
        >>> response = provider.generate_response([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    config = config or {}
    
    provider_name_lower = provider_name.lower()
    
    if provider_name_lower == "openai":
        logger.info("Initializing OpenAI provider")
        return OpenAIProvider(
            model=config.get("model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
        )
    
    # Future provider implementations:
    # elif provider_name_lower == "anthropic":
    #     from .providers import AnthropicProvider
    #     return AnthropicProvider(...)
    #
    # elif provider_name_lower == "gemini":
    #     from .providers import GeminiProvider
    #     return GeminiProvider(...)
    #
    # elif provider_name_lower == "local":
    #     from .providers import LocalProvider
    #     return LocalProvider(...)
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: openai (more coming soon)"
        )