from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'model-settings', views.ModelSettingsViewSet)
router.register(r'job-analysis', views.JobAnalysisViewSet, basename='job-analysis')
router.register(r'api-keys', views.APIKeyViewSet)

urlpatterns = [
    path('', include(router.urls)),
] 