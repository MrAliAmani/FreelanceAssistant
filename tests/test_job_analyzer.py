"""Test job analysis functionality."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from src.models import OllamaEmbedding, OpenRouterModel
from src.analysis import JobAnalyzer

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

@pytest.fixture(scope="module")
def job_post():
    """Load job post from file."""
    job_post_path = Path(__file__).parent.parent / 'utils' / 'job_post.md'
    with open(job_post_path, 'r', encoding='utf-8') as f:
        return f.read()

@pytest.fixture(scope="module")
def results(job_post):
    """Run analysis once for all tests."""
    print("\nRunning job analysis with OpenRouter model...")
    analyzer = JobAnalyzer(
        embedding_model=OllamaEmbedding(),
        inference_model=OpenRouterModel(api_key=OPENROUTER_API_KEY)
    )
    results = analyzer.analyze_job_post(job_post)
    print("Analysis complete.")
    return results

def test_structure(results):
    """Test result structure."""
    assert isinstance(results, dict)
    expected_keys = [
        'jobAnalysis',
        'clientCharacteristics',
        'approachAnalysis',
        'solutionAnalysis',
        'questionsAnalysis',
        'proposal',
        'prompt'
    ]
    for key in expected_keys:
        assert key in results

def test_content(results):
    """Test content validity."""
    for key, value in results.items():
        assert isinstance(value, str)
        assert len(value) > 0

def test_keywords(results):
    """Test keyword presence."""
    job_analysis = results['jobAnalysis'].lower()
    keywords = ["machine learning", "python", "tensorflow", "pytorch", "data"]
    for keyword in keywords:
        assert keyword in job_analysis

def test_proposal(results):
    """Test proposal length."""
    assert len(results['proposal'].split()) <= 150

def test_empty_input():
    """Test empty input handling."""
    analyzer = JobAnalyzer(
        embedding_model=OllamaEmbedding(),
        inference_model=OpenRouterModel(api_key=OPENROUTER_API_KEY)
    )
    with pytest.raises(ValueError):
        analyzer.analyze_job_post("") 