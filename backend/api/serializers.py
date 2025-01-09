from rest_framework import serializers
from .models import ModelSettings, APIKey

class ModelSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelSettings
        fields = [
            'id', 'name', 'embedding_model', 'inference_model',
            'embedding_class', 'inference_class', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

class APIKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = APIKey
        fields = ['id', 'name', 'service', 'key', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']
        extra_kwargs = {
            'key': {'write_only': True}  # Don't expose API keys in responses
        } 