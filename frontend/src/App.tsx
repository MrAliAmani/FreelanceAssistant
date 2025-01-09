import React, { useState } from 'react';
import { analyzeJobPost } from './services/api';

interface AnalysisResults {
  jobAnalysis: string;
  clientCharacteristics: string;
  approachAnalysis: string;
  solutionAnalysis: string;
  questionsAnalysis: string;
  proposal: string;
  prompt: string;
}

const App: React.FC = () => {
  const [jobPost, setJobPost] = useState<string>('');
  const [embeddingModel, setEmbeddingModel] = useState<string>('nomic-embed-text:latest');
  const [inferenceModel, setInferenceModel] = useState<string>('meta-llama/llama-3.1-405b-instruct:free');
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [activeTab, setActiveTab] = useState<keyof AnalysisResults>('jobAnalysis');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!jobPost.trim()) {
      setError('Please enter a job post');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await analyzeJobPost({
        jobPost,
        embeddingModel,
        inferenceModel,
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze job post');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">Proposal Assistant</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div>
            <label className="block text-sm font-medium text-gray-700">Embedding Model</label>
            <select
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm"
            >
              <option value="nomic-embed-text:latest">Ollama Embedding (Default)</option>
              <option value="text-embedding-3-large">OpenAI Text Embedding 3 (Azure)</option>
              <option value="text-embedding-004">Gemini Embedding</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Inference Model</label>
            <select
              value={inferenceModel}
              onChange={(e) => setInferenceModel(e.target.value)}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm"
            >
              <option value="meta-llama/llama-3.1-405b-instruct:free">OpenRouter Llama 3.1 (Default)</option>
              <option value="gpt-4o">Azure GPT-4</option>
              <option value="Phi-3.5-MoE-instruct">Azure Phi-3.5 (128k)</option>
              <option value="Llama-3.3-70B-Instruct">Azure Llama 3.3</option>
              <option value="Meta-Llama-3.1-405B-Instruct">Azure Meta Llama 3.1</option>
              <option value="Mistral-large-2411">Azure Mistral</option>
              <option value="gemini-2.0-flash-exp">Gemini Flash</option>
              <option value="llama-3.3-70b-versatile">Groq Llama 3.3</option>
              <option value="deepseek/deepseek-chat">DeepSeek Chat</option>
              <option value="deepseek-coder-v2:latest">Ollama DeepSeek</option>
            </select>
          </div>
        </div>

        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700">Job Post</label>
          <textarea
            value={jobPost}
            onChange={(e) => setJobPost(e.target.value)}
            className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm"
            rows={6}
            placeholder="Paste the job post here..."
          />
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-400 text-red-700 rounded-md">
            {error}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className={`w-full py-2 px-4 rounded-md transition-colors ${
            isLoading
              ? 'bg-blue-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          } text-white`}
        >
          {isLoading ? 'Analyzing...' : 'Generate Proposal'}
        </button>

        {results && (
          <div className="mt-8">
            <div className="flex space-x-4 border-b border-gray-200 overflow-x-auto">
              {(Object.keys(results) as Array<keyof AnalysisResults>).map((key) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`py-2 px-4 text-sm font-medium whitespace-nowrap ${
                    activeTab === key
                      ? 'border-b-2 border-blue-600 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </button>
              ))}
            </div>

            <div className="mt-4 p-4 bg-white border border-gray-200 rounded-md shadow-sm">
              <pre className="whitespace-pre-wrap text-sm text-gray-700">{results[activeTab]}</pre>
            </div>

            {activeTab === 'proposal' && (
              <button
                onClick={() => {
                  navigator.clipboard.writeText(results.proposal);
                  // You could add a toast notification here
                }}
                className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors"
              >
                Copy Proposal
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;