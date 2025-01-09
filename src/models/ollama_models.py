import logging
from typing import List, Dict, Any
import requests
from langchain_ollama import OllamaEmbeddings
from .base import BaseModel

logger = logging.getLogger(__name__)

class OllamaEmbedding(BaseModel):
    """Ollama embedding model wrapper"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        """Initialize Ollama embedding model"""
        super().__init__()
        self.base_url = base_url
        logger.info(f"Initialized OllamaEmbedding with base URL: {base_url}")
        self.client = OllamaEmbeddings(
            base_url=base_url,
            model="nomic-embed-text:latest"  # Use nomic-embed-text for embeddings
        )

    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            # Test by embedding a simple sentence
            self.embed(["Test connection to Ollama server"])
            return True
        except Exception as e:
            logger.error(f"Ollama embedding connection test failed: {str(e)}")
            return False

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        try:
            embeddings = self.client.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Ollama embedding failed: {str(e)}")
            raise

class OllamaInference(BaseModel):
    """Ollama inference model wrapper"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        """Initialize Ollama inference model"""
        super().__init__()
        self.base_url = base_url
        self.model = "deepseek-coder-v2:latest"  # Use deepseek-coder-v2 for inference
        logger.info(f"Initialized OllamaInference with base URL: {base_url} and model {self.model}")

    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            # Test by generating a simple response
            response = self.complete(
                messages=[{
                    "role": "user",
                    "content": "Say 'test successful' if you can read this."
                }]
            )
            return "test successful" in response["message"]["content"].lower()
        except Exception as e:
            logger.error(f"Ollama inference connection test failed: {str(e)}")
            return False

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Complete a conversation using Ollama"""
        try:
            # Format messages for Ollama API
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return {
                "message": {
                    "role": "assistant",
                    "content": result["response"]
                }
            }
        except Exception as e:
            logger.error(f"Ollama completion failed: {str(e)}")
            raise 