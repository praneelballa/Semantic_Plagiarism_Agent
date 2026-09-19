import React from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

interface CodeResultProps {
  result: {
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

export const CodeASTDiff: React.FC<CodeResultProps> = ({ result }) => {
  return (
    <div className="p-6 rounded-2xl bg-dark-800/60 border border-slate-800 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400">Classification</span>
          <h4 className="text-sm font-bold text-slate-100">{result.match_type}</h4>
        </div>
        {result.variable_renaming_detected ? (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-950/40 border border-red-500/40 text-red-400 text-xs font-mono font-semibold">
            <AlertCircle className="w-3.5 h-3.5" /> Variable Renaming Detected
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/40 text-emerald-400 text-xs font-mono font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" /> No Identifier Camouflage
          </div>
        )}
      </div>

      <p className="text-xs text-slate-400 font-sans leading-relaxed">{result.evidence_note}</p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center font-mono">
        <div className="p-3 rounded-lg bg-dark-900 border border-slate-800">
          <span className="text-[10px] text-slate-400 block uppercase">AST Structural</span>
          <span className="text-sm font-bold text-cyan">{(result.structural_similarity * 100).toFixed(1)}%</span>
        </div>
        <div className="p-3 rounded-lg bg-dark-900 border border-slate-800">
          <span className="text-[10px] text-slate-400 block uppercase">UniXcoder</span>
          <span className="text-sm font-bold text-slate-100">{(result.semantic_similarity * 100).toFixed(1)}%</span>
        </div>
        <div className="p-3 rounded-lg bg-dark-900 border border-slate-800">
          <span className="text-[10px] text-slate-400 block uppercase">Lexical Overlap</span>
          <span className="text-sm font-bold text-slate-100">{(result.raw_lexical_similarity * 100).toFixed(1)}%</span>
        </div>
        <div className="p-3 rounded-lg bg-dark-900 border border-slate-800">
          <span className="text-[10px] text-slate-400 block uppercase">Canonical Diff</span>
          <span className="text-sm font-bold text-slate-100">{(result.canonical_code_similarity * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Canonical AST (Reference)</span>
          <pre className="p-3 rounded-lg bg-dark-900 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto h-48">
            <code>{result.canonical_source}</code>
          </pre>
        </div>
        <div className="space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Canonical AST (Suspect)</span>
          <pre className="p-3 rounded-lg bg-dark-900 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto h-48">
            <code>{result.canonical_suspect}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
