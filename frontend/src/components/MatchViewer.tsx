import React from 'react';
import { MatchDetail } from '../api';
import { LanguageBadge } from './LanguageBadge';

interface MatchViewerProps {
  matches: MatchDetail[];
}

export const MatchViewer: React.FC<MatchViewerProps> = ({ matches }) => {
  if (matches.length === 0) {
    return (
      <div className="p-8 text-center rounded-xl bg-dark-800/40 border border-slate-800 text-xs text-slate-500 font-mono">
        No matches exceeded detection thresholds.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {matches.map((m, idx) => {
        const isCross = m.submitted_lang.toLowerCase() !== m.source_lang.toLowerCase();
        return (
          <div key={idx} className="p-5 rounded-xl bg-dark-800/60 border border-slate-800 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-cyan/10 border border-cyan/30 text-cyan text-[11px] font-mono font-bold">
                  MATCH #{idx + 1}
                </span>
                <span className="text-xs font-semibold text-slate-300">{m.match_type}</span>
              </div>
              <div className="flex items-center gap-2">
                <LanguageBadge lang={m.submitted_lang} isCrossLingual={isCross} />
                <span className="text-slate-600 font-mono">?</span>
                <LanguageBadge lang={m.source_lang} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                  <span>SUBMITTED PASSAGE</span>
                  <span>PAGE {m.submitted_page}</span>
                </div>
                <div className="p-3 rounded-lg bg-dark-900/80 border border-slate-800 text-xs text-slate-200 leading-relaxed font-sans border-l-2 border-l-cyan">
                  {m.submitted_text}
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                  <span>SOURCE REFERENCE ({m.source_document})</span>
                  <span>PAGE {m.source_page}</span>
                </div>
                <div className="p-3 rounded-lg bg-dark-900/80 border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans border-l-2 border-l-emerald-400">
                  {m.source_text}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-[11px] font-mono pt-1 text-slate-400">
              <span>Semantic Similarity: <strong className="text-slate-200">{(m.semantic_similarity * 100).toFixed(1)}%</strong></span>
              <span>Lexical Overlap: <strong className="text-slate-200">{(m.lexical_similarity * 100).toFixed(1)}%</strong></span>
              {m.final_confidence_score !== undefined && (
                <span>Composite Confidence: <strong className="text-cyan">{(m.final_confidence_score * 100).toFixed(1)}%</strong></span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
