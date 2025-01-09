import os
import pytest
from src.models import (
    OpenRouterModel,  # Default inference model
    OllamaEmbedding,  # Default embedding model
    AzureGPT4,
    AzureEmbedding,
    AzurePhi35,
    AzureLlama33,
    AzureMetaLlama31,
    AzureMistral,
    GeminiEmbedding,
    GeminiInference,
    GroqModel,
    DeepSeekModel,
    OllamaDeepseek
)
import requests.exceptions
from pytest import fail

# Configure default timeout for all tests
pytestmark = pytest.mark.timeout(60)

# Fixtures for different API tokens
@pytest.fixture
def openrouter_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return key

@pytest.fixture
def github_token():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN environment variable not set")
    return token

@pytest.fixture
def gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return key

@pytest.fixture
def groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return key

@pytest.fixture
def azure_endpoint():
    return "https://models.inference.ai.azure.com"

@pytest.fixture
def deepseek_openrouter_api_key():
    key = os.environ.get("DEEPSEEK_OPENROUTER_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_OPENROUTER_API_KEY environment variable not set")
    return key

# Default model tests first
def test_openrouter_connection(openrouter_api_key):
    """Test connection to meta-llama/llama-3.1-405b-instruct:free (Default Inference Model)"""
    try:
        model = OpenRouterModel(api_key=openrouter_api_key)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_ollama_embedding_connection():
    """Test connection to Ollama embedding model (Default Embedding Model)"""
    try:
        model = OllamaEmbedding()
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

# Azure model tests
def test_gpt4o_connection(github_token, azure_endpoint):
    """Test connection to GPT-4o model"""
    try:
        model = AzureGPT4(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_azure_embedding_connection(github_token, azure_endpoint):
    """Test connection to Text Embedding 3 model"""
    try:
        model = AzureEmbedding(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_phi35_connection(github_token, azure_endpoint):
    """Test connection to Phi-3.5-MoE instruct model"""
    try:
        model = AzurePhi35(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_llama33_connection(github_token, azure_endpoint):
    """Test connection to Llama-3.3-70B-Instruct model"""
    try:
        model = AzureLlama33(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_meta_llama31_connection(github_token, azure_endpoint):
    """Test connection to Meta-Llama-3.1-405B-Instruct model"""
    try:
        model = AzureMetaLlama31(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_mistral_connection(github_token, azure_endpoint):
    """Test connection to Mistral Large 24.11 model"""
    try:
        model = AzureMistral(token=github_token, endpoint=azure_endpoint)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

# Other external model tests
def test_gemini_embedding_connection(gemini_api_key):
    """Test connection to Gemini text-embedding-004 model"""
    try:
        model = GeminiEmbedding(api_key=gemini_api_key)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_gemini_inference_connection(gemini_api_key):
    """Test connection to Gemini-2.0-flash-exp model"""
    try:
        model = GeminiInference(api_key=gemini_api_key)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_groq_connection(groq_api_key):
    """Test connection to llama-3.3-70b-versatile"""
    try:
        model = GroqModel(api_key=groq_api_key)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_deepseek_connection(deepseek_openrouter_api_key):
    """Test connection to deepseek/deepseek-chat model"""
    try:
        model = DeepSeekModel(api_key=deepseek_openrouter_api_key)
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}")

def test_ollama_deepseek_connection():
    """Test connection to Ollama deepseek model"""
    try:
        model = OllamaDeepseek()
        assert model.test_connection()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        fail("Connection timeout or error")
    except Exception as e:
        fail(f"Test failed: {str(e)}") 