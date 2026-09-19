import React from 'react';
import { ScoreBadge } from './ScoreBadge';
import { AlertTriangle, ShieldCheck, AlertCircle } from 'lucide-react';

interface RiskOverviewProps {
  coveragePct: number;
  riskBand: string;
  totalChunks: number;
  matchedChunks: number;
  avgConfidence: number;
}

export const RiskOverview: React.FC<RiskOverviewProps> = ({
  coveragePct,
  riskBand,
  totalChunks,
  matchedChunks,
  avgConfidence,
}) => {
  const getRiskDetails = (band: string) => {
    if (band.toLowerCase().includes('incidental')) {
      return {
        icon: ShieldCheck,
        style: 'bg-emerald-950/30 border-emerald-500/30 text-emerald-400',
      };
    }
    if (band.toLowerCase().includes('moderate')) {
      return {
        icon: AlertTriangle,
        style: 'bg-amber-950/30 border-amber-500/30 text-amber-400',
      };
    }
    return {
      icon: AlertCircle,
      style: 'bg-red-950/30 border-red-500/30 text-red-400',
    };
  };

  const { icon: RiskIcon, style: riskStyle } = getRiskDetails(riskBand);

  return (
    <div className="p-6 rounded-2xl bg-dark-800/60 border border-slate-800 space-y-6">
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-6">
          <ScoreBadge score={coveragePct} />
          <div className="space-y-1">
            <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400">Risk Assessment</span>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-semibold ${riskStyle}`}>
              <RiskIcon className="w-4 h-4" />
              <span>{riskBand}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 w-full md:w-auto">
          <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800/80 text-center">
            <span className="text-slate-400 text-[10px] font-mono uppercase block">Total Segments</span>
            <span className="text-lg font-bold text-slate-100">{totalChunks}</span>
          </div>
          <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800/80 text-center">
            <span className="text-slate-400 text-[10px] font-mono uppercase block">Matches Flagged</span>
            <span className="text-lg font-bold text-cyan">{matchedChunks}</span>
          </div>
          <div className="p-3 rounded-xl bg-dark-900/60 border border-slate-800/80 text-center">
            <span className="text-slate-400 text-[10px] font-mono uppercase block">Confidence</span>
            <span className="text-lg font-bold text-slate-100">{avgConfidence.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
