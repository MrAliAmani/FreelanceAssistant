import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

interface JobPostAnalysisRequest {
  jobPost: string;
  embeddingModel: string;
  inferenceModel: string;
}

interface JobPostAnalysisResponse {
  jobAnalysis: string;
  clientCharacteristics: string;
  approachAnalysis: string;
  solutionAnalysis: string;
  questionsAnalysis: string;
  proposal: string;
  prompt: string;
}

export const analyzeJobPost = async (request: JobPostAnalysisRequest): Promise<JobPostAnalysisResponse> => {
  try {
    const response = await axios.post<JobPostAnalysisResponse>(`${API_BASE_URL}/job-analysis/`, {
      job_post: request.jobPost,
      embedding_model: request.embeddingModel,
      inference_model: request.inferenceModel
    });
    return response.data;
  } catch (error) {
    const err = error as any;
    throw new Error(err?.response?.data?.error || 'Failed to analyze job post');
  }
}; 