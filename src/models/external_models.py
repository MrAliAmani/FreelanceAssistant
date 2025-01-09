import requests
import json
import time
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import logging

from .base import BaseModel

class GeminiBase(BaseModel):
    """Base class for Gemini models"""
    def __init__(self, api_key: str):
        self.api_key = api_key

class GeminiEmbedding(GeminiBase):
    """Class for Gemini text-embedding-004 model"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key,
            task_type="retrieval_document"
        )
    
    def test_connection(self) -> bool:
        try:
            result = self.model.embed_query("Test sentence for embedding.")
            return len(result) > 0
        except Exception:
            return False

class GeminiInference(GeminiBase):
    """Class for Gemini-2.0-flash-exp model"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = GoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=1.0,
            convert_system_message_to_human=True,
            max_tokens=50
        )
    
    def test_connection(self) -> bool:
        try:
            response = self.model.invoke([
                HumanMessage(content="Say 'test successful' if you can read this.")
            ])
            return len(response.content) > 0
        except Exception:
            return False

class OpenRouterModel:
    """OpenRouter model for inference"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = self  # Make the model itself act as the client

    def complete(self, messages: list, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 1000, model: str = "meta-llama/llama-3.1-405b-instruct:free") -> dict:
        """Complete a chat conversation"""
        try:
            # Format messages for OpenRouter
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Make API call to OpenRouter
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/your-username/your-repo",
                    "X-Title": "FreelanceAssistant"
                },
                json={
                    "model": model,
                    "messages": formatted_messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens
                },
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Ensure the response has the expected structure
            if "choices" not in result or not result["choices"]:
                raise ValueError("Invalid response from OpenRouter: missing choices")
            if "message" not in result["choices"][0]:
                raise ValueError("Invalid response from OpenRouter: missing message")
            
            return result
        except Exception as e:
            logging.error(f"OpenRouter completion failed: {str(e)}")
            # Return a properly structured error response
            return {
                "choices": [{
                    "message": {
                        "content": f"Error: {str(e)}"
                    }
                }]
            }

    def test_connection(self) -> bool:
        """Test connection to OpenRouter"""
        try:
            response = self.complete(
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": "Say 'Connection successful' if you can read this."
                }],
                max_tokens=10
            )
            return "choices" in response and len(response["choices"]) > 0
        except Exception as e:
            logging.error(f"OpenRouter connection test failed: {str(e)}")
            return False

class GroqModel(BaseModel):
    """Class for Groq models"""
    def __init__(self, api_key: str):
        self.model = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=1.0,
            max_tokens=50,
            top_p=1.0
        )
    
    def test_connection(self) -> bool:
        try:
            response = self.model.invoke([
                HumanMessage(content="Say 'test successful' if you can read this.")
            ])
            return len(response.content) > 0
        except Exception:
            return False

class DeepSeekModel(BaseModel):
    """Class for DeepSeek models using OpenRouter"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-chat"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://freelanceassistant.ai",
            "X-Title": "FreelanceAssistant",
            "Content-Type": "application/json",
            "X-Model": self.model
        }
    
    def test_connection(self) -> bool:
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Say 'test successful' if you can read this."}],
                        "max_tokens": 50,
                        "temperature": 1.0
                    },
                    timeout=30  # 30 seconds timeout
                )
                
                if response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    return False
                
                response.raise_for_status()
                result = response.json()
                return len(result.get("choices", [])) > 0 and len(result["choices"][0].get("message", {}).get("content", "")) > 0
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return False
            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError):
                return False 