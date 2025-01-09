from typing import Dict
import time
from ..models.base import BaseModel
from ..models import OllamaEmbedding, OllamaInference

class JobAnalyzer:
    def __init__(self, embedding_model: BaseModel = None, inference_model: BaseModel = None):
        """Initialize with OllamaEmbedding and OllamaInference as default models"""
        self.embedding_model = embedding_model or OllamaEmbedding()
        self.inference_model = inference_model or OllamaInference()

    def _analyze_with_retry(self, prompt: str, job_post: str, max_retries: int = 3, delay: float = 1.0) -> str:
        """Analyze with retry logic for failed attempts"""
        for attempt in range(max_retries):
            try:
                response = self.inference_model.complete(
                    messages=[{
                        "role": "system",
                        "content": "You are a professional freelancer analyzing job posts and writing proposals."
                    }, {
                        "role": "user",
                        "content": f"{prompt}\n\nJob Post:\n{job_post}"
                    }],
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=1000
                )
                return response["message"]["content"]
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    return f"Analysis failed after {max_retries} attempts: {str(e)}"
                time.sleep(delay * (attempt + 1))  # Exponential backoff

    def analyze_job_post(self, job_post: str) -> Dict[str, str]:
        """Analyze a job post using specified models and return structured analysis"""
        if not job_post.strip():
            raise ValueError("Job post cannot be empty")

        # Get embeddings using Ollama's format
        embeddings = self.embedding_model.embed([job_post])

        # Prepare analysis prompts with improved context
        analysis_prompts = {
            'jobAnalysis': '''Analyze the key requirements, responsibilities, and scope of this job post. 
                          Focus on technical requirements, experience needed, and project scope.
                          Format the response in clear sections with bullet points.''',
            'clientCharacteristics': '''Analyze the client preferences, company culture, and project characteristics from this job post. 
                                   Focus on work environment, values, and expectations.
                                   Format the response with clear headings and bullet points.''',
            'approachAnalysis': '''Suggest a detailed approach and methodology for this project based on the job post. 
                               Include specific steps, best practices, and timeline estimates.
                               Format as a numbered list with sub-points.''',
            'solutionAnalysis': '''Recommend specific technical solutions, tools, and technologies for this project.
                               Include justification for each recommendation.
                               Format with clear categories and bullet points.''',
            'questionsAnalysis': '''What are the key questions to ask and points to address in the proposal?
                                Format as a numbered list of questions with brief explanations.''',
            'proposal': '''Write a concise, professional proposal (max 150 words) for this job post.
                       Start with an attention-grabbing first sentence.
                       First paragraph about client and problem.
                       Second paragraph about solution and approach.
                       Include trust-building elements and competitive advantages.
                       Make a recommendation for extra value.
                       End with a call to action.
                       Format with clear paragraphs and professional tone.''',
        }

        results = {}
        for key, prompt in analysis_prompts.items():
            # Use retry for all sections since we're using Ollama
            results[key] = self._analyze_with_retry(prompt, job_post)

        # Add the prompt used for reference
        results['prompt'] = str(analysis_prompts)
        return results
