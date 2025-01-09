from setuptools import setup, find_packages

setup(
    name="freelance_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'django',
        'django-cors-headers',
        'azure-ai-inference',
        'langchain-google-genai',
        'langchain-groq',
        'langchain-core',
        'requests',
    ],
) 