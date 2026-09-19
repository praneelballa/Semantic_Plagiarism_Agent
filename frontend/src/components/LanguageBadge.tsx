import React from 'react';

interface LanguageBadgeProps {
  lang: string;
  isCrossLingual?: boolean;
}

export const LanguageBadge: React.FC<LanguageBadgeProps> = ({ lang, isCrossLingual = false }) => {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider font-semibold border ${
        isCrossLingual
          ? 'bg-purple-950/40 border-purple-500/40 text-purple-300'
          : 'bg-cyan/10 border-cyan/30 text-cyan'
      }`}
    >
      {lang}
      {isCrossLingual && <span className="ml-1 text-[9px] font-normal text-purple-400">(Cross-Lingual)</span>}
    </span>
  );
};
