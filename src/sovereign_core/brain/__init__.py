"""Brain module - Speech-to-text and LLM processing."""

from .llm_factory import get_llm_provider
from .llm_interface import LLMProvider

__all__ = ["LLMProvider", "get_llm_provider"]