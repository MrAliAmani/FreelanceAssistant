from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings

from .models import ModelSettings, APIKey
from .serializers import ModelSettingsSerializer, APIKeySerializer
from src.models import OllamaEmbedding, OllamaInference
from src.analysis import JobAnalyzer

class ModelSettingsViewSet(viewsets.ModelViewSet):
    """
    ViewSet for model settings with default models:
    - Embedding: OllamaEmbedding (nomic-embed-text:latest)
    - Inference: OllamaInference (deepseek-coder-v2:latest)
    """
    queryset = ModelSettings.objects.all()
    serializer_class = ModelSettingsSerializer

    def get_queryset(self):
        return ModelSettings.objects.filter(is_active=True)

    @action(detail=False, methods=['get'])
    def default_settings(self, request):
        """Get default model settings"""
        return Response({
            'embedding_model': 'nomic-embed-text:latest',
            'inference_model': 'deepseek-coder-v2:latest',
            'embedding_class': 'OllamaEmbedding',
            'inference_class': 'OllamaInference',
        })

class JobAnalysisViewSet(viewsets.ViewSet):
    """ViewSet for job post analysis using default models"""

    def create(self, request):
        """Analyze a job post"""
        job_post = request.data.get('job_post')
        if not job_post:
            return Response(
                {'error': 'Job post is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Initialize analyzer with default models
            analyzer = JobAnalyzer(
                embedding_model=OllamaEmbedding(),
                inference_model=OllamaInference()
            )

            # Analyze job post
            results = analyzer.analyze_job_post(job_post)
            return Response(results)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class APIKeyViewSet(viewsets.ModelViewSet):
    """ViewSet for managing API keys"""
    queryset = APIKey.objects.all()
    serializer_class = APIKeySerializer

    def get_queryset(self):
        return APIKey.objects.filter(is_active=True)
