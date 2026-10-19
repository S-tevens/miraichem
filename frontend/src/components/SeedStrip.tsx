import { useMemo, useState } from "react";
import { log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import type { Config } from "../types";
import "./SeedStrip.css";

interface Props {
  config: Config;
  chemAcc: number;
  hfError: number | null;
}

/** One dot per random seed on a log error axis, with the mean, the spread and the two baselines. */
export function SeedStrip({ config, chemAcc, hfError }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const lo = Math.min(...config.errors) * 0.5;
  const hi = Math.max(hfError ?? 0, ...config.errors, config.mean + config.std) * 1.6;
  const domain: [number, number] = [
    10 ** Math.max(-2, Math.floor(Math.log10(Math.max(lo, 0.01)))),
    10 ** Math.ceil(Math.log10(hi)),
  ];
  const left = 12;
  const right = 14;
  const x = log(domain, [left, width - right]);
  const rowTop = 34;
  const rowH = 78;
  const height = rowTop + rowH + 40;
  const cy = rowTop + rowH / 2;

  // Deterministic vertical jitter so overlapping seeds stay visible.
  const jitter = useMemo(
    () => config.errors.map((_, i) => ((i * 37) % 11) / 10 - 0.5),
    [config.errors],
  );
  const ticks = logTicks(domain).filter((t) => [1, 10, 100, 1000].includes(t) || t === domain[0]);
  const spreadLo = Math.max(domain[0], config.mean - config.std);
  const spreadHi = config.mean + config.std;

  return (
    <div ref={ref} className="strip">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Errors of ${config.n_seeds} seeds, mean ${mha(config.mean)} mHa, worst ${mha(config.worst)} mHa`}
      >
        <rect x={x(domain[0])} y={rowTop} width={x(chemAcc) - x(domain[0])} height={rowH} className="strip__band" />
        <text x={x(domain[0]) + 6} y={rowTop - 10} className="strip__ref strip__ref--ok">
          chemical accuracy
        </text>
        {hfError !== null && (
          <>
            <line x1={x(hfError)} x2={x(hfError)} y1={rowTop - 6} y2={rowTop + rowH} className="strip__hf" />
            <text x={x(hfError) - 6} y={rowTop - 10} textAnchor="end" className="strip__ref strip__ref--hf">
              Hartree-Fock {mha(hfError)}
            </text>
          </>
        )}
        <line x1={x(spreadLo)} x2={x(spreadHi)} y1={cy} y2={cy} className="strip__spread" />
        {config.errors.map((e, i) => (
          <circle
            key={i}
            cx={x(e)}
            cy={cy + jitter[i] * (rowH - 22)}
            r={hover === i ? 8 : 6}
            className={`strip__dot ${e < chemAcc ? "strip__dot--ok" : ""}`}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        <path
          d={`M ${x(config.mean)} ${cy - 11} l 7 11 l -7 11 l -7 -11 z`}
          className="strip__mean"
          aria-label={`mean ${mha(config.mean)} mHa`}
        />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={rowTop + rowH + 4} y2={rowTop + rowH + 9} className="strip__tick" />
            <text x={x(t)} y={rowTop + rowH + 26} textAnchor="middle" className="strip__axis num">
              {t}
            </text>
          </g>
        ))}
      </svg>
      {hover !== null && (
        <div
          className="tip"
          style={{ left: x(config.errors[hover]), top: cy + jitter[hover] * (rowH - 22) }}
        >
          <strong>Seed {hover + 1}</strong>
          <br />
          {mha(config.errors[hover])} mHa
        </div>
      )}
      <p className="strip__legend small">
        <span className="strip__key strip__key--dot" /> one random seed
        <span className="strip__key strip__key--ok" /> within chemical accuracy
        <span className="strip__key strip__key--mean" /> mean, with ±1 standard deviation
      </p>
    </div>
  );
}
