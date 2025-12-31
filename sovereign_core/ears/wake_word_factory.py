"""
Factory for creating wake word detection provider instances.

This module provides a factory function that instantiates the appropriate
wake word provider based on configuration, enabling zero-friction engine swapping.
"""

import logging
from typing import Any, Dict

from .wake_word_interface import WakeWordProvider
from .providers import PorcupineProvider

logger = logging.getLogger(__name__)


def get_wake_word_provider(
    provider_name: str = "porcupine",
    config: Dict[str, Any] | None = None,
) -> WakeWordProvider:
    """
    Factory function to get the appropriate wake word detection provider instance.
    
    This function instantiates the correct provider based on the provider name
    and configuration. Adding new providers requires only updating this factory
    and implementing the new provider class.
    
    Args:
        provider_name: Name of the provider ("porcupine", future: "snowboy", "precise", etc.)
        config: Optional configuration dictionary with provider-specific settings.
                If None, uses default values.
    
    Returns:
        WakeWordProvider: Instance of the requested provider
        
    Raises:
        ValueError: If provider_name is not recognized or required config is missing
        
    Example:
        >>> config = {
        ...     "access_key": "your-porcupine-key",
        ...     "keywords": ["hey sovereign"],
        ...     "sensitivity": 0.7
        ... }
        >>> provider = get_wake_word_provider("porcupine", config)
        >>> for detection in provider.start():
        ...     print(f"Wake word detected at {detection.timestamp}")
    """
    config = config or {}
    
    provider_name_lower = provider_name.lower()
    
    if provider_name_lower == "porcupine":
        logger.info("Initializing Porcupine wake word provider")
        
        access_key = config.get("access_key")
        if not access_key:
            raise ValueError(
                "Porcupine provider requires 'access_key' in config. "
                "Get your key from https://console.picovoice.ai/"
            )
        
        # Keywords are optional if custom keyword_path is provided
        keywords = config.get("keywords")
        keyword_path = config.get("keyword_path")
        
        if not keywords and not keyword_path:
            raise ValueError(
                "Porcupine provider requires either 'keywords' or 'keyword_path' in config. "
                "Specify built-in keywords (e.g., ['picovoice']) or path to custom .ppn file."
            )
        
        return PorcupineProvider(
            access_key=access_key,
            keywords=keywords,
            sensitivity=config.get("sensitivity", 0.5),
            keyword_path=keyword_path,
        )
    
    # Future provider implementations:
    # elif provider_name_lower == "snowboy":
    #     from .providers import SnowboyProvider
    #     return SnowboyProvider(
    #         model_path=config.get("model_path"),
    #         sensitivity=config.get("sensitivity", 0.5),
    #     )
    #
    # elif provider_name_lower == "precise":
    #     from .providers import PreciseProvider
    #     return PreciseProvider(
    #         model_path=config.get("model_path"),
    #         threshold=config.get("threshold", 0.5),
    #     )
    #
    # elif provider_name_lower == "custom":
    #     from .providers import CustomProvider
    #     return CustomProvider(**config)
    
    else:
        raise ValueError(
            f"Unknown wake word provider: {provider_name}. "
            f"Supported providers: porcupine (more coming soon)"
        )