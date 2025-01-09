FAILED tests/test_model_connections.py::test_gemini_embedding_connection - assert False
FAILED tests/test_model_connections.py::test_gemini_inference_connection - assert False
FAILED tests/test_model_connections.py::test_deepseek_connection - assert False










Make .env with api keys that switch.
run tests and handle rate limits.
To manage multiple API keys with rate limits effectively, follow this organized plan:

1. Store Multiple API Keys in a Single .env File
In your .env file, store each API key as a separate environment variable:

Copy
openrouter_api_1="key1"
openrouter_api_2="key2"
openrouter_api_3="key3"
2. Load API Keys into a Deque
Use Python's dotenv library to load the keys and store them in a deque for easy rotation:

python
Copy
from dotenv import load_dotenv
import os
from collections import deque

load_dotenv()
keys = [
    os.getenv('openrouter_api_1'),
    os.getenv('openrouter_api_2'),
    os.getenv('openrouter_api_3')
]
key_queue = deque(keys)
3. Implement API Request Function
Create a function to make requests and check for rate limits:

python
Copy
import requests

def make_request(url, key):
    headers = {'Authorization': f'Bearer {key}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        return None
    return response
4. Rotate API Keys on Rate Limit
Use a loop to rotate through keys when a rate limit is encountered:

python
Copy
max_attempts = len(keys)
attempts = 0

while attempts < max_attempts:
    key = key_queue[0]
    response = make_request(url, key)
    if response is not None:
        break
    else:
        key_queue.rotate(-1)
        attempts += 1
        # Implement exponential backoff or wait based on 'Retry-After' header
        time.sleep(60)  # Example delay

if attempts >= max_attempts:
    print("All API keys are rate-limited. Unable to proceed.")
5. Handle All Keys Rate-Limited
Ensure your application handles the scenario where all keys are rate-limited:

Implement exponential backoff.

Check for 'Retry-After' headers in 429 responses.

Inform the user or log the event.

6. Security and Best Practices
Ensure the .env file is not exposed or committed to version control.

Handle exceptions and errors gracefully.

Log API usage and errors for monitoring.

7. Testing
Write unit tests to simulate rate limits and verify key switching:

python
Copy
import unittest
from unittest.mock import patch
from your_module import make_request, key_queue

class TestApiKeyRotation(unittest.TestCase):
    @patch('requests.get')
    def test_key_rotation_on_rate_limit(self, mock_get):
        mock_get.return_value.status_code = 429
        for _ in range(len(key_queue)):
            response = make_request(url, key_queue[0])
            self.assertIsNone(response)
            key_queue.rotate(-1)
        # Add assertions for all keys being rate-limited
8. Documentation
Document your code to explain the key management logic.

Provide guidelines for adding or removing API keys.

By following this plan, you can efficiently manage multiple API keys, handle rate limits, and ensure your application remains reliable and secure.








ANY OLLAMA model for inference







D:\Projects\FreelanceAssistant\src\models\ollama_models.py:17: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaEmbeddings``.
  self.client = OllamaEmbeddings(







Send the results of previous prompts for proposal generation or summarise and send them. make a workflow.

