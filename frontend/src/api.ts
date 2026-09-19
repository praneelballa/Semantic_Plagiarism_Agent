import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1';

export interface MatchDetail {
  submitted_chunk_id: string;
  source_chunk_id: string;
  submitted_text: string;
  source_text: string;
  submitted_page: number;
  source_page: number;
  submitted_lang: string;
  source_lang: string;
  source_document: string;
  bi_encoder_score?: number;
  cross_encoder_score?: number | null;
  final_confidence_score?: number;
  semantic_similarity: number;
  lexical_similarity: number;
  match_type: string;
  timestamp_reference?: string | null;
}

export interface AnalysisResponse {
  analysis_id: string;
  modality: 'text' | 'audio' | 'code';
  total_chunks?: number;
  matched_chunks?: number;
  plagiarism_coverage_pct?: number;
  risk_band?: string;
  reranker_enabled?: boolean;
  matches?: MatchDetail[];
  result?: {
    syntax_valid: boolean;
    semantic_similarity: number;
    structural_similarity: number;
    canonical_code_similarity: number;
    raw_lexical_similarity: number;
    variable_renaming_detected: boolean;
    match_type: string;
    evidence_note: string;
    canonical_suspect: string;
    canonical_source: string;
  };
}

export interface HealthResponse {
  status: string;
  environment: string;
  project: string;
}

export const checkHealth = async (): Promise<HealthResponse> => {
  const res = await axios.get(`${API_BASE}/health`);
  return res.data;
};

export const ingestCorpusFiles = async (files: File[]): Promise<{ documents_added: number; chunks_added: number; index_size: number }> => {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await axios.post(`${API_BASE}/corpus/ingest`, formData);
  return res.data;
};

export const runAnalysis = async (
  file: File,
  modality: 'text' | 'audio' | 'code' = 'text',
  sourceCodeRef?: string
): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', modality);
  formData.append('compare_corpus', 'true');
  if (sourceCodeRef) {
    formData.append('source_code_reference', sourceCodeRef);
  }
  const res = await axios.post(`${API_BASE}/analyze`, formData);
  return res.data;
};

export const fetchAnalysisById = async (analysisId: string): Promise<AnalysisResponse> => {
  const res = await axios.get(`${API_BASE}/analyze/${analysisId}`);
  return res.data;
};
