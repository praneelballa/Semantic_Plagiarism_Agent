import React from 'react';
import { MatchDetail } from '../api';
import { Volume2, Clock } from 'lucide-react';

interface AudioMatchProps {
  matches: MatchDetail[];
}

export const AudioMatch: React.FC<AudioMatchProps> = ({ matches }) => {
  return (
    <div className="space-y-4">
      {matches.map((m, idx) => (
        <div key={idx} className="p-5 rounded-xl bg-dark-800/60 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-cyan text-xs font-mono">
              <Volume2 className="w-4 h-4" />
              <span>Audio Citation #{idx + 1}</span>
            </div>
            {m.timestamp_reference && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-dark-900 border border-slate-800 text-[11px] font-mono text-cyan">
                <Clock className="w-3.5 h-3.5" />
                <span>{m.timestamp_reference}</span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-dark-900/80 border border-slate-800 text-xs text-slate-200 leading-relaxed border-l-2 border-l-cyan">
              <span className="text-[10px] font-mono text-slate-400 block mb-1">TRANSCRIBED SPEECH</span>
              {m.submitted_text}
            </div>
            <div className="p-3 rounded-lg bg-dark-900/80 border border-slate-800 text-xs text-slate-300 leading-relaxed border-l-2 border-l-emerald-400">
              <span className="text-[10px] font-mono text-slate-400 block mb-1">MATCHED SOURCE ({m.source_document})</span>
              {m.source_text}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
