# Freelance Assistant

A tool to assist in writing proposals for Upwork job posts using various AI models. The application analyzes job posts and generates tailored proposals using a combination of embedding and inference models.

## Features

- Job post analysis with multiple aspects:
  - Key requirements and responsibilities analysis
  - Client characteristics and company culture analysis
  - Project approach and methodology suggestions
  - Technical solutions and tools recommendations
  - Key questions for proposal
  - Concise, professional proposal generation (max 150 words)

- Multiple AI Models Support:
  - **Default Embedding Model**: Ollama Embedding (nomic-embed-text:latest)
  - **Default Inference Model**: OpenRouter Llama 3.1 (meta-llama/llama-3.1-405b-instruct:free)
  - Additional models from Azure, Gemini, Groq, and more

## Prerequisites

- Python 3.12+
- Node.js 18+
- Ollama running locally
- API keys for various services:
  - OpenRouter API key
  - Azure (GitHub token)
  - Gemini API key
  - Groq API key
  - DeepSeek OpenRouter API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FreelanceAssistant
```

2. Set up the Python environment:
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Create a `.env` file in the project root with your API keys:
```env
OPENROUTER_API_KEY=your_openrouter_key
GITHUB_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
DEEPSEEK_OPENROUTER_API_KEY=your_deepseek_key
```

## Running the Application

1. Start Ollama (required for embedding model):
```bash
ollama serve
```

2. Set up and start the Django backend (in a new terminal):
```bash
cd backend

# Initialize the database
python manage.py makemigrations
python manage.py migrate

# Create a superuser for API access
python manage.py createsuperuser
# Follow the prompts to create your admin account

# Create an API key for OpenRouter
python manage.py shell
>>> from api.models import APIKey
>>> APIKey.objects.create(
...     name='default',
...     service='openrouter',
...     key='your_openrouter_api_key',
...     is_active=True
... )
>>> exit()

# Start the backend server
python manage.py runserver
```

3. Start the React frontend (in another terminal):
```bash
cd frontend
npm start
```

4. Access the application:
- Open http://localhost:3000 in your browser
- The backend API will be available at http://localhost:8000
- Admin interface at http://localhost:8000/admin

## Usage

1. Open the application in your browser
2. Select your preferred embedding and inference models (defaults are recommended)
3. Paste your job post in the text area
4. Click "Generate Proposal"
5. View the analysis results in different tabs:
   - Job Analysis
   - Client Characteristics
   - Approach Analysis
   - Solution Analysis
   - Questions Analysis
   - Final Proposal
6. Copy the generated proposal or any analysis section

## Troubleshooting

1. If you get a "Failed to analyze job post" error:
   - Check that Ollama is running (`ollama serve`)
   - Verify your OpenRouter API key is correctly set in the database
   - Check the Django server logs for specific error messages

2. If models fail to connect:
   - Verify all API keys are correctly set in `.env`
   - Check that Ollama has the required models installed:
     ```bash
     ollama pull nomic-embed-text:latest
     ollama pull deepseek-coder:latest
     ```

3. Common issues:
   - 403 Forbidden: Make sure you've created and set up the API keys in Django admin
   - Connection errors: Check if all services (Ollama, Django, React) are running
   - Model errors: Verify API keys and model availability

## Development

The project follows a test-driven development (TDD) approach:
- Tests are in the `tests/` directory
- Run tests with: `pytest tests/`
- Frontend tests: `cd frontend && npm test`

## Project Structure

```
FreelanceAssistant/
├── backend/             # Django backend
│   ├── api/            # REST API
│   └── backend/        # Django settings
├── frontend/           # React frontend
│   ├── src/           # Source code
│   └── public/        # Static files
├── src/               # Python package
│   ├── models/        # AI model implementations
│   └── analysis/      # Analysis logic
├── tests/             # Test files
└── utils/             # Utility files
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
