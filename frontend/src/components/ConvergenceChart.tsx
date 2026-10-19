import { useMemo, useState } from "react";
import { linear, linearTicks, log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { Seg } from "../ui/ui";
import { CHEM_ACC_MHA } from "../types";
import "./charts.css";

interface Props {
  trace: number[]; // total energy of every evaluation (Hartree)
  unmitigated?: number[];
  eExact: number | null;
  eHf: number | null;
}

type View = "error" | "energy";

const toPath = (pts: [number, number][]) =>
  pts.map(([x, y], i) => `${i ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");

/** Energy or error against the number of energy evaluations, with the best value so far. */
export function ConvergenceChart({ trace, unmitigated = [], eExact, eHf }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [view, setView] = useState<View>(eExact === null ? "energy" : "error");
  const [hover, setHover] = useState<number | null>(null);

  const n = trace.length;
  const left = 54;
  const right = 14;
  const height = 250;
  const top = 14;
  const bottom = 34;

  const err = (e: number) => Math.max(Math.abs(e - (eExact ?? 0)) * 1000, 1e-3);
  const series = useMemo(() => {
    const val = (e: number) => (view === "error" ? err(e) : e);
    const best: number[] = [];
    let b = Infinity;
    for (const e of trace) {
      b = Math.min(b, view === "error" ? err(e) : e);
      best.push(b);
    }
    return {
      raw: trace.map(val),
      best,
      unmit: unmitigated.map(val),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, unmitigated, view, eExact]);

  if (n < 2) return <p className="small muted">No convergence data was saved for this run.</p>;

  const all = [...series.raw, ...series.unmit];
  const x = linear([0, n - 1], [left, Math.max(left + 120, width - right)]);

  let y: (v: number) => number;
  let yTicks: number[];
  let fmt: (v: number) => string;
  if (view === "error") {
    const lo = Math.max(1e-3, Math.min(...series.best) * 0.6);
    const hi = Math.max(...all, eHf !== null && eExact !== null ? Math.abs(eHf - eExact) * 1000 : 0) * 1.3;
    const dom: [number, number] = [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))];
    y = log(dom, [height - bottom, top]);
    yTicks = logTicks(dom).filter((t) => [0.01, 0.1, 1, 10, 100, 1000, 10000].includes(t));
    fmt = (v) => `${v}`;
  } else {
    const lo = Math.min(...all, eExact ?? Infinity, eHf ?? Infinity);
    const hi = Math.max(...all, eExact ?? -Infinity);
    const pad = (hi - lo) * 0.06 || 0.01;
    y = linear([lo - pad, hi + pad], [height - bottom, top]);
    yTicks = linearTicks(lo, hi, 4);
    fmt = (v) => v.toFixed(3);
  }

  const line = (vals: number[]) => toPath(vals.map((v, i) => [x(i), y(v)] as [number, number]));
  const hfErr = eHf !== null && eExact !== null ? Math.abs(eHf - eExact) * 1000 : null;
  const xTicks = linearTicks(0, n - 1, 5);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left - left) / (x(n - 1) - x(0))) * (n - 1));
    setHover(Math.min(n - 1, Math.max(0, i)));
  };

  return (
    <div ref={ref} className="chart">
      <div className="chart__tools">
        <Seg
          label="Convergence view"
          value={view}
          onChange={setView}
          options={[
            ...(eExact !== null ? [{ id: "error" as const, label: "Error vs exact" }] : []),
            { id: "energy" as const, label: "Energy" },
          ]}
        />
      </div>
      <svg
        width={width}
        height={height}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label={`Convergence over ${n} evaluations; final ${view === "error" ? mha(err(trace[n - 1])) + " mHa error" : trace[n - 1].toFixed(5) + " Hartree"}`}
      >
        {view === "error" && (
          <rect
            x={x(0)}
            y={y(CHEM_ACC_MHA)}
            width={x(n - 1) - x(0)}
            height={Math.max(0, y(1e-3) - y(CHEM_ACC_MHA))}
            className="chart__band"
          />
        )}
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={left} x2={x(n - 1)} y1={y(t)} y2={y(t)} className="chart__grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">
              {fmt(t)}
            </text>
          </g>
        ))}
        {view === "error" && hfErr !== null && (
          <line x1={x(0)} x2={x(n - 1)} y1={y(hfErr)} y2={y(hfErr)} className="chart__hf" />
        )}
        {view === "energy" && eExact !== null && (
          <line x1={x(0)} x2={x(n - 1)} y1={y(eExact)} y2={y(eExact)} className="chart__exact" />
        )}
        {view === "energy" && eHf !== null && (
          <line x1={x(0)} x2={x(n - 1)} y1={y(eHf)} y2={y(eHf)} className="chart__hf" />
        )}
        {series.unmit.length > 1 && <path d={line(series.unmit)} className="chart__line chart__line--dim" />}
        <path d={line(series.raw)} className="chart__line chart__line--raw" />
        <path d={line(series.best)} className="chart__line chart__line--vqe" />
        {hover !== null && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1={top} y2={height - bottom} className="chart__cross" />
            <circle cx={x(hover)} cy={y(series.raw[hover])} r={4} className="chart__pt" />
          </>
        )}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={height - 14} textAnchor="middle" className="chart__axis num">
            {t}
          </text>
        ))}
        <text x={x(n - 1)} y={height - 1} textAnchor="end" className="chart__axis">
          energy evaluations
        </text>
        <text x={left} y={height - 1} className="chart__axis">
          {view === "error" ? "error (mHa)" : "energy (Hartree)"}
        </text>
      </svg>
      {hover !== null && (
        <div className="tip num" style={{ left: Math.min(Math.max(x(hover), 90), width - 90), top: y(series.raw[hover]) }}>
          <strong>Evaluation {hover}</strong>
          <br />
          {view === "error" ? `${mha(series.raw[hover])} mHa` : `${trace[hover].toFixed(5)} Ha`}
        </div>
      )}
      <div className="chart__key small">
        <span><i className="sw sw--vqe" /> best so far</span>
        <span><i className="sw sw--raw" /> each evaluation</span>
        {series.unmit.length > 1 && <span><i className="sw sw--dim" /> before mitigation</span>}
        {view === "error" && hfErr !== null && <span><i className="sw sw--hf" /> Hartree-Fock</span>}
        {view === "error" && <span><i className="sw sw--ok" /> within chemical accuracy</span>}
        {view === "energy" && eExact !== null && <span><i className="sw sw--exact" /> exact</span>}
      </div>
    </div>
  );
}
