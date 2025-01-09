"""Model implementations for the application."""

from .base import BaseModel
from .external_models import OpenRouterModel
from .ollama_models import OllamaEmbedding, OllamaInference

__all__ = [
    'BaseModel',
    'OpenRouterModel',
    'OllamaEmbedding',
    'OllamaInference',
] 