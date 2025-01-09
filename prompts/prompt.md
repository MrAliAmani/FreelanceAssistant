I want to create a project to assist me in writing proposal for upwork job posts.
Here is the description:
I will use different models for each part of the project.
I will adopt the test driven development approach (TDD).
write tests in tests folder including unit tests and integration tests if needed.
then develop the minimum required code to pass the tests.
have a simple ui using react for seeing the results and choose the models.
user must see an option to choose the embedding model and also an option to choose the inference model.
langchain and langgraph can be used for models. use the langchain except for openrouter models.
the environment variables are set in .env file.





embedding models:
OpenAI Text Embedding 3 (large) from github tokens (default)
text-embedding-004 from gemini_api_key
nomic-embed-text:latest from ollama like 

inference models:
OpenAI GPT-4o from github tokens (default)
Phi-3.5-MoE instruct (128k) from github tokens
Llama-3.3-70B-Instruct from github tokens
Meta-Llama-3.1-405B-Instruct from github tokens
Mistral Large 24.11 from github tokens
gemini-2.0-flash-exp from gemini_api_key
meta-llama/llama-3.1-405b-instruct:free from openrouter
llama-3.3-70b-versatile from groq api key
deepseek/deepseek-chat from DEEPSEEK_OPENROUTER_API_KEY
deepseek-coder-v2:latest from ollama 




we will do the project in steps:
1. get the job post as input from the user.
2. analyze the job post with a prompt and using the model.
use the selected models for embedding and inference in ui. analyze the job post using the model and show the result in the related tab in ui.
3. analyze client characteristics using the model.
4. analyze my approach like these:
thorough testing
tailored to project requirements
5. analyze the solution, techniques and tools:
libraries like selenium, tensorflow, etc.
machine learning methods and models like cnn, rnn, yolo, etc.
generative ai models like groq, llama, phi-3.5, etc.
automation methods
backend like django
frontend like react
etc.
6.analyze the related questions and/or call to action for the proposal.
7. write a recommended proposal based on these instructions and examples and the results from steps 2 to 6:
Write a proposal and be consice. the first sentece should grab attention. the first paragraph is about client and problem. the second one is about solution and approach. we  can use trust making sentences and competitive advantage in the proposal. make a recommendation in the second paragraph for the extra mile service. make a call to action at last.
be concise (less than 150 words). only write in plain text and may be one emoji to be distinctive. steer away from cliche and be differentiated from other freelancers. stay professional. don't make a lot of claims.



UI
use react for a user friendly and responsive ui. be minimalist for this app. have a selection for the inference models and another one for embedding models. 
there should be a box for the user to input his job post.
then the user should be able to see the result of each of the steps. these outputs should be in separate tabs. 
at last show the prompt. 
the user is able to copy the final proposal.



* use the embedding model (text-embedding-3-large) provided as needed. 
OpenAI Text Embedding 3 (large):
import os

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "text-embedding-3-large"
token = os.environ["GITHUB_TOKEN"]

client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token)
)

response = client.embed(
    input=["first phrase", "second phrase", "third phrase"],
    model=model_name
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)






OpenAI GPT-4o:
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)




phi 3.5 instruct
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Phi-3.5-MoE-instruct"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)





Llama-3.3-70B-Instruct
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Llama-3.3-70B-Instruct"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)





Meta-Llama-3.1-405B-Instruct
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Meta-Llama-3.1-405B-Instruct"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)





Mistral Large 24.11
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Mistral-large-2411"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)





