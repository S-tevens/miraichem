import { log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { CHEM_ACC_MHA } from "../types";
import "./charts.css";

interface Props {
  low: number;
  mid: number;
  high: number;
  hf: number;
  /** The error that was actually measured, once the run has finished. */
  actual?: number | null;
}

/** The predicted error and its 80% interval on a log axis, against Hartree-Fock and chemical accuracy. */
export function ForecastChart({ low, mid, high, hf, actual }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const vals = [low, high, hf, CHEM_ACC_MHA, ...(actual ? [actual] : [])];
  const dom: [number, number] = [
    10 ** Math.floor(Math.log10(Math.min(...vals) * 0.7)),
    10 ** Math.ceil(Math.log10(Math.max(...vals) * 1.4)),
  ];
  const left = 12;
  const right = 14;
  const x = log(dom, [left, Math.max(left + 160, width - right)]);
  const height = 112;
  const cy = 52;
  const ticks = logTicks(dom).filter((t) => [0.01, 0.1, 1, 10, 100, 1000, 10000].includes(t));
  const inside = actual != null && actual >= low && actual <= high;

  return (
    <div ref={ref} className="chart">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Predicted error ${mha(mid)} mHa, 80% interval ${mha(low)} to ${mha(high)} mHa`}
      >
        <rect x={x(dom[0])} y={cy - 18} width={x(CHEM_ACC_MHA) - x(dom[0])} height={36} className="chart__band" />
        <line x1={x(dom[0])} x2={x(dom[1])} y1={cy} y2={cy} className="chart__grid" />
        <line x1={x(hf)} x2={x(hf)} y1={cy - 22} y2={cy + 22} className="chart__hf" />
        <text x={x(hf)} y={cy - 28} textAnchor="middle" className="chart__axis">
          Hartree-Fock
        </text>
        <line x1={x(low)} x2={x(high)} y1={cy} y2={cy} className="forecast__range" />
        <line x1={x(low)} x2={x(low)} y1={cy - 7} y2={cy + 7} className="forecast__range" />
        <line x1={x(high)} x2={x(high)} y1={cy - 7} y2={cy + 7} className="forecast__range" />
        <circle cx={x(mid)} cy={cy} r={7.5} className="forecast__mid" />
        {actual != null && (
          <g>
            <path d={`M ${x(actual)} ${cy + 12} l 7 11 l -14 0 z`} className={inside ? "forecast__hit" : "forecast__miss"} />
            <text x={x(actual)} y={cy + 38} textAnchor="middle" className="chart__axis">
              measured {mha(actual)}
            </text>
          </g>
        )}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={height - 2} textAnchor="middle" className="chart__axis num">
            {t}
          </text>
        ))}
      </svg>
      <div className="chart__key small">
        <span><i className="dotkey dotkey--front" /> predicted {mha(mid)} mHa</span>
        <span><i className="sw sw--hf" /> Hartree-Fock ({mha(hf)})</span>
        <span><i className="sw sw--ok" /> within chemical accuracy</span>
        {actual != null && <span>{inside ? "measured value fell inside the interval" : "measured value fell outside the interval"}</span>}
      </div>
    </div>
  );
}
