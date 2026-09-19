import React from 'react';

interface ScoreBadgeProps {
  score: number;
  size?: number;
}

export const ScoreBadge: React.FC<ScoreBadgeProps> = ({ score, size = 120 }) => {
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const getColor = (val: number) => {
    if (val <= 25) return '#10b981'; // Green
    if (val <= 50) return '#f59e0b'; // Amber
    return '#ef4444';                // Red
  };

  const strokeColor = getColor(score);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#1e293b"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          fill="transparent"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-2xl font-black tracking-tight" style={{ color: strokeColor }}>
          {score.toFixed(1)}%
        </span>
        <span className="text-[10px] uppercase font-mono tracking-widest text-slate-400">Coverage</span>
      </div>
    </div>
  );
};
