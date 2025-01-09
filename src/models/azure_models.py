from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import numpy as np
import time
import logging
from requests.exceptions import RequestException, Timeout

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .base import BaseModel

class AzureModelBase(BaseModel):
    """Base class for Azure models"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        self.token = token
        self.endpoint = endpoint
        self.timeout = timeout
        self.credential = AzureKeyCredential(token)
        logger.info(f"Initialized {self.__class__.__name__} with endpoint: {self.endpoint}")

class AzureGPT4(AzureModelBase):
    """Class for GPT-4o model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("ChatCompletionsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        try:
            logger.info("Testing connection to GPT-4o model...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'test successful' if you can read this.")
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=50,
                model="gpt-4o"
            )
            success = len(response.choices[0].message.content) > 0
            logger.info(f"GPT-4o connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"GPT-4o connection test failed with error: {str(e)}")
            return False

class AzureEmbedding(AzureModelBase):
    """Class for Text Embedding 3 model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = EmbeddingsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("EmbeddingsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Testing connection to Text Embedding 3 model (attempt {attempt + 1}/{max_retries})...")
                response = self.client.embed(
                    input=["Test sentence for embedding."],
                    model="text-embedding-3-large"
                )
                success = len(response.data[0].embedding) > 0
                logger.info(f"Embedding connection test {'successful' if success else 'failed'}")
                return success
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:  # Rate limit
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                logger.error(f"Embedding connection test failed with error: {str(e)}")
                return False

class AzurePhi35(AzureModelBase):
    """Class for Phi-3.5-MoE instruct model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("ChatCompletionsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        try:
            logger.info("Testing connection to Phi-3.5 model...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'test successful' if you can read this.")
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=50,
                model="Phi-3.5-MoE-instruct"
            )
            success = len(response.choices[0].message.content) > 0
            logger.info(f"Phi-3.5 connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Phi-3.5 connection test failed with error: {str(e)}")
            return False

class AzureLlama33(AzureModelBase):
    """Class for Llama-3.3-70B-Instruct model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("ChatCompletionsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        try:
            logger.info("Testing connection to Llama-3.3 model...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'test successful' if you can read this.")
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=50,
                model="Llama-3.3-70B-Instruct"
            )
            success = len(response.choices[0].message.content) > 0
            logger.info(f"Llama-3.3 connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Llama-3.3 connection test failed with error: {str(e)}")
            return False

class AzureMetaLlama31(AzureModelBase):
    """Class for Meta-Llama-3.1-405B-Instruct model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("ChatCompletionsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        try:
            logger.info("Testing connection to Meta-Llama-3.1 model...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'test successful' if you can read this.")
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=50,
                model="Meta-Llama-3.1-405B-Instruct"
            )
            success = len(response.choices[0].message.content) > 0
            logger.info(f"Meta-Llama-3.1 connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Meta-Llama-3.1 connection test failed with error: {str(e)}")
            return False

class AzureMistral(AzureModelBase):
    """Class for Mistral Large 24.11 model"""
    def __init__(self, token: str, endpoint: str = "https://models.inference.ai.azure.com", timeout: int = 30):
        super().__init__(token, endpoint, timeout)
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            logger.info("ChatCompletionsClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsClient: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        try:
            logger.info("Testing connection to Mistral model...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'test successful' if you can read this.")
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=50,
                model="Mistral-large-2411"
            )
            success = len(response.choices[0].message.content) > 0
            logger.info(f"Mistral connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Mistral connection test failed with error: {str(e)}")
            return False 