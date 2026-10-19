import { useState } from "react";
import { deviceName, mha, shortName } from "../lib/format";
import { log, logTicks } from "../lib/scale";
import { useWidth } from "../lib/hooks";
import { CHEM_ACC_MHA, type RankedRow } from "../types";
import "./charts.css";
import "./ParetoChart.css";

export type CostAxis = "gates" | "shots";

interface Props {
  rows: RankedRow[];
  axis: CostAxis;
  bestHash: string;
  onSelect: (hash: string) => void;
}

const costOf = (r: RankedRow, axis: CostAxis): number =>
  axis === "shots" ? r.total_shots ?? 0 : r.gates ?? r.depth ?? 0;

/** Error against cost for every run: the front of non-dominated runs is highlighted. */
export function ParetoChart({ rows, axis, bestHash, onSelect }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<RankedRow | null>(null);

  const pts = rows.filter((r) => r.error_mha !== null && costOf(r, axis) > 0);
  if (!pts.length) return <p className="small muted">Nothing to plot for this axis.</p>;

  const left = 52;
  const right = 16;
  const top = 14;
  const bottom = 38;
  const height = 300;
  const costs = pts.map((r) => costOf(r, axis));
  const errs = pts.map((r) => Math.max(r.error_mha as number, 1e-3));
  const cx: [number, number] = [10 ** Math.floor(Math.log10(Math.min(...costs) * 0.9)), 10 ** Math.ceil(Math.log10(Math.max(...costs) * 1.1))];
  const cy: [number, number] = [10 ** Math.floor(Math.log10(Math.min(...errs) * 0.8)), 10 ** Math.ceil(Math.log10(Math.max(...errs) * 1.2))];
  const x = log(cx, [left, Math.max(left + 160, width - right)]);
  const y = log(cy, [height - bottom, top]);
  const xt = logTicks(cx).filter((t) => [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9].includes(t));
  const yt = logTicks(cy).filter((t) => [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000].includes(t));
  const front = pts.filter((r) => (axis === "shots" ? r.pareto_shots : r.pareto_gates));
  const frontSorted = [...front].sort((a, b) => costOf(a, axis) - costOf(b, axis));
  const step = frontSorted.map((r) => `${x(costOf(r, axis)).toFixed(1)},${y(Math.max(r.error_mha as number, 1e-3)).toFixed(1)}`);

  const label = axis === "shots" ? "total shots" : pts[0].backend === "ideal" ? "circuit depth" : "two-qubit gates";
  const tipX = hover ? x(costOf(hover, axis)) : 0;
  const tipY = hover ? y(Math.max(hover.error_mha as number, 1e-3)) : 0;

  return (
    <div ref={ref} className="chart pareto">
      <svg width={width} height={height} role="img" aria-label={`Error against ${label} for ${pts.length} runs`}>
        <rect
          x={x(cx[0])}
          y={y(CHEM_ACC_MHA)}
          width={x(cx[1]) - x(cx[0])}
          height={Math.max(0, y(cy[0]) - y(CHEM_ACC_MHA))}
          className="chart__band"
        />
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={left} x2={x(cx[1])} y1={y(t)} y2={y(t)} className="chart__grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">
              {t}
            </text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={`x${t}`} x={x(t)} y={height - 18} textAnchor="middle" className="chart__axis num">
            {t >= 1e6 ? `${t / 1e6}M` : t >= 1e3 ? `${t / 1e3}k` : t}
          </text>
        ))}
        {step.length > 1 && <polyline points={step.join(" ")} className="pareto__front" />}
        {pts.map((r) => {
          const onFront = axis === "shots" ? r.pareto_shots : r.pareto_gates;
          const best = r.hash === bestHash;
          return (
            <circle
              key={r.hash}
              cx={x(costOf(r, axis))}
              cy={y(Math.max(r.error_mha as number, 1e-3))}
              r={best ? 7 : onFront ? 5.5 : 4}
              className={`pareto__dot ${onFront ? "pareto__dot--front" : ""} ${best ? "pareto__dot--best" : ""}`}
              tabIndex={0}
              role="button"
              aria-label={`${shortName(r)}: ${mha(r.error_mha)} mHa`}
              onMouseEnter={() => setHover(r)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(r)}
              onBlur={() => setHover(null)}
              onClick={() => onSelect(r.hash)}
              onKeyDown={(e) => e.key === "Enter" && onSelect(r.hash)}
            />
          );
        })}
        <text x={x(cx[1])} y={height - 2} textAnchor="end" className="chart__axis">
          {label} (log scale)
        </text>
        <text x={left} y={height - 2} className="chart__axis">
          error (mHa)
        </text>
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(Math.max(tipX, 110), width - 110), top: tipY }}>
          <strong>{shortName(hover)}</strong>
          <br />
          {hover.backend === "noisy" ? `${deviceName(hover.device)} · ` : ""}
          {mha(hover.error_mha)} mHa · {costOf(hover, axis).toLocaleString("en")} {axis === "shots" ? "shots" : "gates"}
        </div>
      )}
      <div className="chart__key small">
        <span><i className="dotkey dotkey--front" /> on the Pareto front</span>
        <span><i className="dotkey" /> other runs</span>
        <span><i className="dotkey dotkey--best" /> recommended</span>
        <span><i className="sw sw--ok" /> within chemical accuracy</span>
      </div>
    </div>
  );
}
