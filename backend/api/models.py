from django.db import models
from django.conf import settings

class ModelSettings(models.Model):
    """Model settings for the application"""
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
    DEFAULT_INFERENCE_MODEL = "meta-llama/llama-3.1-405b-instruct:free"
    DEFAULT_EMBEDDING_CLASS = "OllamaEmbedding"
    DEFAULT_INFERENCE_CLASS = "OpenRouterModel"

    name = models.CharField(max_length=100, unique=True)
    embedding_model = models.CharField(max_length=100, default=DEFAULT_EMBEDDING_MODEL)
    inference_model = models.CharField(max_length=100, default=DEFAULT_INFERENCE_MODEL)
    embedding_class = models.CharField(max_length=100, default=DEFAULT_EMBEDDING_CLASS)
    inference_class = models.CharField(max_length=100, default=DEFAULT_INFERENCE_CLASS)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Model Settings"
        verbose_name_plural = "Model Settings"

    def __str__(self):
        return self.name

class APIKey(models.Model):
    """API keys for different services"""
    name = models.CharField(max_length=100)
    key = models.CharField(max_length=500)
    service = models.CharField(max_length=100)  # e.g., 'openrouter', 'azure', etc.
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"
        unique_together = ('name', 'service')

    def __str__(self):
        return f"{self.service} - {self.name}"
