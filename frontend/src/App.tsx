import React, { useState, useEffect } from 'react';
import { runAnalysis, ingestCorpusFiles, checkHealth, AnalysisResponse } from './api';
import { RiskOverview } from './components/RiskOverview';
import { MatchViewer } from './components/MatchViewer';
import { CodeASTDiff } from './components/CodeASTDiff';
import { AudioMatch } from './components/AudioMatch';
import { UploadPanel } from './components/UploadPanel';
import { ShieldCheck, Database, FileSpreadsheet, Activity } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'audit' | 'corpus'>('audit');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<string>('Checking...');
  const [corpusSuccess, setCorpusSuccess] = useState<string | null>(null);

  useEffect(() => {
    checkHealth()
      .then(d => setHealthStatus(d.status))
      .catch(() => setHealthStatus('Offline'));
  }, []);

  const handleAnalyze = async (file: File, modality: 'text' | 'audio' | 'code', codeRef?: string) => {
    setIsLoading(true);
    try {
      const res = await runAnalysis(file, modality, codeRef);
      setAnalysisResult(res);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Analysis request failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCorpusUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setIsLoading(true);
      try {
        const files = Array.from(e.target.files);
        const res = await ingestCorpusFiles(files);
        setCorpusSuccess(`Added ${res.documents_added} documents (${res.chunks_added} chunks). Corpus size: ${res.index_size}`);
      } catch (err: any) {
        alert(err.response?.data?.detail || 'Corpus ingestion failed.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen pb-16">
      <header className="border-b border-slate-800 bg-dark-800/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan/10 border border-cyan/30 text-cyan">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-slate-100">
                PLAGIARISM AGENT
              </h1>
              <span className="text-[10px] font-mono text-cyan tracking-wider uppercase block">
                Multimodal / Cross-Lingual
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-dark-900 px-3 py-1.5 rounded-full border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span>API: {healthStatus.toUpperCase()}</span>
            </div>
            <nav className="flex gap-2">
              <button
                onClick={() => setActiveTab('audit')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium ${activeTab === 'audit' ? 'bg-cyan text-dark-900 font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Audit Workspace
              </button>
              <button
                onClick={() => setActiveTab('corpus')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium ${activeTab === 'corpus' ? 'bg-cyan text-dark-900 font-bold' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Corpus Storage
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 pt-8 space-y-8">
        {activeTab === 'audit' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-1">
              <UploadPanel onAnalyze={handleAnalyze} isLoading={isLoading} />
            </div>

            <div className="lg:col-span-2 space-y-6">
              {analysisResult ? (
                <>
                  {analysisResult.modality !== 'code' && (
                    <RiskOverview
                      coveragePct={analysisResult.plagiarism_coverage_pct || 0}
                      riskBand={analysisResult.risk_band || 'Unknown'}
                      totalChunks={analysisResult.total_chunks || 0}
                      matchedChunks={analysisResult.matched_chunks || 0}
                      avgConfidence={
                        analysisResult.matches && analysisResult.matches.length > 0
                          ? analysisResult.matches.reduce((acc, m) => acc + (m.final_confidence_score ?? m.semantic_similarity), 0) / analysisResult.matches.length
                          : 0
                      }
                    />
                  )}

                  <div className="space-y-4">
                    <h3 className="text-sm font-mono uppercase tracking-wider text-slate-400">
                      Detection Evidence
                    </h3>

                    {analysisResult.modality === 'code' && analysisResult.result ? (
                      <CodeASTDiff result={analysisResult.result} />
                    ) : analysisResult.modality === 'audio' && analysisResult.matches ? (
                      <AudioMatch matches={analysisResult.matches} />
                    ) : (
                      <MatchViewer matches={analysisResult.matches || []} />
                    )}
                  </div>
                </>
              ) : (
                <div className="p-12 text-center rounded-2xl bg-dark-800/40 border border-slate-800 space-y-3">
                  <FileSpreadsheet className="w-10 h-10 text-slate-600 mx-auto" />
                  <h4 className="text-slate-300 font-bold text-sm">No Active Submission</h4>
                  <p className="text-xs text-slate-500 max-w-sm mx-auto">
                    Select a document, speech file, or Python script to generate an audit report.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto p-8 rounded-2xl bg-dark-800/60 border border-slate-800 space-y-6">
            <div className="flex items-center gap-3 text-cyan">
              <Database className="w-6 h-6" />
              <h2 className="text-lg font-bold text-slate-100">Index Reference Materials</h2>
            </div>
            <p className="text-xs text-slate-400">
              Upload institutional textbooks, course syllabi, or academic journals to add them to the FAISS vector database.
            </p>
            <input
              type="file"
              multiple
              onChange={handleCorpusUpload}
              className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-cyan file:text-dark-900 hover:file:bg-cyan-400 cursor-pointer"
            />
            {corpusSuccess && (
              <div className="p-4 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
                {corpusSuccess}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
