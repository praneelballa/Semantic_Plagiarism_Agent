import React, { useState } from 'react';
import { Upload, FileText, Mic, Code2, Loader2 } from 'lucide-react';

interface UploadPanelProps {
  onAnalyze: (file: File, modality: 'text' | 'audio' | 'code', codeRef?: string) => Promise<void>;
  isLoading: boolean;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({ onAnalyze, isLoading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modality, setModality] = useState<'text' | 'audio' | 'code'>('text');
  const [codeRef, setCodeRef] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    onAnalyze(selectedFile, modality, modality === 'code' ? codeRef : undefined);
  };

  return (
    <form onSubmit={handleSubmit} className="p-6 rounded-2xl bg-dark-800/60 border border-slate-800 space-y-5">
      <div className="space-y-1">
        <h2 className="text-sm font-bold text-slate-100 tracking-tight">Audit Submission</h2>
        <p className="text-xs text-slate-400">Select submission modality and inspect for cross-lingual plagiarism.</p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <button
          type="button"
          onClick={() => setModality('text')}
          className={`flex items-center justify-center gap-2 p-2.5 rounded-xl border text-xs font-medium transition-all ${
            modality === 'text'
              ? 'bg-cyan/10 border-cyan text-cyan'
              : 'bg-dark-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <FileText className="w-4 h-4" /> Text
        </button>
        <button
          type="button"
          onClick={() => setModality('audio')}
          className={`flex items-center justify-center gap-2 p-2.5 rounded-xl border text-xs font-medium transition-all ${
            modality === 'audio'
              ? 'bg-cyan/10 border-cyan text-cyan'
              : 'bg-dark-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <Mic className="w-4 h-4" /> Audio
        </button>
        <button
          type="button"
          onClick={() => setModality('code')}
          className={`flex items-center justify-center gap-2 p-2.5 rounded-xl border text-xs font-medium transition-all ${
            modality === 'code'
              ? 'bg-cyan/10 border-cyan text-cyan'
              : 'bg-dark-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
          }`}
        >
          <Code2 className="w-4 h-4" /> Code
        </button>
      </div>

      <div className="relative border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-slate-700 transition-colors bg-dark-900/30">
        <input
          type="file"
          id="file-upload"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          accept={
            modality === 'audio'
              ? '.wav,.mp3'
              : modality === 'code'
              ? '.py'
              : '.pdf,.docx,.txt,.md'
          }
        />
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-6 h-6 text-cyan" />
          <span className="text-xs font-medium text-slate-300">
            {selectedFile ? selectedFile.name : 'Choose file or drag & drop'}
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            {modality === 'audio' ? 'WAV, MP3' : modality === 'code' ? 'PYTHON (.py)' : 'PDF, DOCX, TXT, MD'}
          </span>
        </div>
      </div>

      {modality === 'code' && (
        <div className="space-y-1">
          <label className="text-[10px] font-mono uppercase text-slate-400">Reference Source Code (Optional)</label>
          <textarea
            value={codeRef}
            onChange={(e) => setCodeRef(e.target.value)}
            placeholder="Paste authoritative source code here for direct AST structural diff..."
            className="w-full h-24 p-3 rounded-lg bg-dark-900 border border-slate-800 text-xs font-mono text-slate-300 focus:outline-none focus:border-cyan/40"
          />
        </div>
      )}

      <button
        type="submit"
        disabled={!selectedFile || isLoading}
        className="w-full py-2.5 px-4 rounded-xl bg-cyan text-dark-900 font-bold text-xs tracking-wider uppercase hover:bg-cyan-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" /> Analyzing Submission...
          </>
        ) : (
          'Execute Semantic Audit'
        )}
      </button>
    </form>
  );
};
