import { useMemo } from "react";
import { log, logTicks, linear } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { Panel } from "../ui/ui";
import { CHEM_ACC_MHA, type SqdGroup, type SqdRow } from "../types";
import "./charts.css";
import "./SqdScaling.css";

interface Pt {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
  note: string;
}
interface Line {
  key: string;
  label: string;
  tone: string;
  pts: Pt[];
}

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};

/** The default setting of the N2 seed study; every size is run the same way so they compare. */
const isStudy = (g: SqdGroup) =>
  g.molecule.toUpperCase().startsWith("N2") &&
  g.noise_model === null &&
  g.readout_noise === 0.02 &&
  g.iterations === 3 &&
  g.samples_per_batch === 300 &&
  g.num_batches === 3 &&
  g.shots === 20000;

function LineChart({ lines, domain, axis, aria, band }: { lines: Line[]; domain: [number, number]; axis: string; aria: string; band?: boolean }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const xs = [...new Set(lines.flatMap((l) => l.pts.map((p) => p.x)))].sort((a, b) => a - b);
  const left = 54;
  const right = 130;
  const top = 14;
  const height = 260;
  const bottom = height - 40;
  const x = linear([xs[0] - 1.5, xs[xs.length - 1] + 1.5], [left, Math.max(left + 160, width - right)]);
  const y = log(domain, [bottom, top]);
  const ticks = logTicks(domain).filter((t) => Math.log10(t) % 1 === 0);
  // end-of-line labels, pushed apart so lines that finish close together stay readable
  const labelYs = new Map<string, number>();
  let prevY = -Infinity;
  for (const e of lines
    .map((l) => {
      const ok = l.pts.filter((p) => Number.isFinite(p.y) && p.y > 0).sort((a, b) => a.x - b.x);
      return { key: l.key, y: ok.length ? y(ok[ok.length - 1].y) : NaN };
    })
    .filter((e) => Number.isFinite(e.y))
    .sort((a, b) => a.y - b.y)) {
    prevY = Math.max(e.y, prevY + 15);
    labelYs.set(e.key, prevY);
  }
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={height} role="img" aria-label={aria}>
        {band && <rect x={left} y={y(CHEM_ACC_MHA)} width={Math.max(0, x(xs[xs.length - 1] + 1.5) - left)} height={Math.max(0, y(domain[0]) - y(CHEM_ACC_MHA))} className="chart__band" />}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={x(xs[xs.length - 1] + 1.5)} y1={y(t)} y2={y(t)} className="chart__grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">{t >= 1000 ? `${t / 1000}k` : t < 1 ? t : t}</text>
          </g>
        ))}
        {xs.map((q) => (
          <text key={q} x={x(q)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{q}</text>
        ))}
        <text x={x(xs[Math.floor(xs.length / 2)])} y={height - 6} textAnchor="middle" className="chart__axis">qubits</text>
        {lines.map((l) => {
          const pts = l.pts.filter((p) => Number.isFinite(p.y) && p.y > 0).sort((a, b) => a.x - b.x);
          if (!pts.length) return null;
          const last = pts[pts.length - 1];
          const labelY = labelYs.get(l.key) ?? y(last.y);
          return (
            <g key={l.key} className={`scl scl--${l.tone}`}>
              {pts.length > 1 && <polyline points={pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")} className="scl__line" />}
              {pts.map((p) => (
                <g key={p.x}>
                  {p.lo !== undefined && p.hi !== undefined && <line x1={x(p.x)} x2={x(p.x)} y1={y(p.lo)} y2={y(p.hi)} className="scl__bar" />}
                  <circle cx={x(p.x)} cy={y(p.y)} r={5} className="scl__dot">
                    <title>{`${l.label}, ${p.x} qubits: ${p.note}`}</title>
                  </circle>
                </g>
              ))}
              <text x={x(last.x) + 12} y={labelY + 4} className="scl__label">{l.label}</text>
            </g>
          );
        })}
        <text x={left} y={top - 2} className="chart__axis">{axis}</text>
      </svg>
    </div>
  );
}

/** Error and time against the number of qubits, from every saved N2 size, all run the same way. */
export function SqdScaling({ rows, groups }: { rows: SqdRow[]; groups: SqdGroup[] }) {
  const data = useMemo(() => {
    const study = groups.filter(isStudy);
    const sizes = [...new Set(study.map((g) => g.qubits ?? 0))].filter(Boolean).sort((a, b) => a - b);
    if (sizes.length < 2) return null;
    const at = (q: number) => study.filter((g) => g.qubits === q);
    const hf: Pt[] = [];
    const cc: Pt[] = [];
    for (const q of sizes) {
      const g = at(q)[0];
      if (g.hf_error_mha) hf.push({ x: q, y: g.hf_error_mha, note: `${mha(g.hf_error_mha)} mHa` });
      if (g.ccsd_error_mha) cc.push({ x: q, y: g.ccsd_error_mha, note: `${mha(g.ccsd_error_mha)} mHa` });
    }
    const layers = (n: number): Pt[] =>
      sizes.flatMap((q) => {
        const g = at(q).find((s) => s.n_reps === n);
        return g
          ? [
              {
                x: q,
                y: g.mean_error_mha,
                lo: Math.max(0.05, g.mean_error_mha - g.std_error_mha),
                hi: g.mean_error_mha + g.std_error_mha,
                note: `${mha(g.mean_error_mha)} ± ${mha(g.std_error_mha)} mHa over ${g.n_seeds} seeds`,
              },
            ]
          : [];
      });
    const times = (pick: (r: SqdRow) => number | null | undefined, filter: (r: SqdRow) => boolean): Pt[] =>
      sizes.flatMap((q) => {
        const v = rows.filter((r) => r.qubits === q && r.status === "ok" && !r.hardware && filter(r)).map(pick).filter((t): t is number => typeof t === "number");
        return v.length ? [{ x: q, y: median(v), note: `${median(v).toFixed(median(v) < 10 ? 2 : 0)} s` }] : [];
      });
    const study1 = (r: SqdRow) => r.readout_noise === 0.02 && r.noise_model === null && r.samples_per_batch === 300 && r.iterations === 3;
    return {
      sizes,
      errors: [
        { key: "hf", label: "Hartree-Fock", tone: "hf", pts: hf },
        { key: "cc", label: "CCSD", tone: "cc", pts: cc },
        { key: "q1", label: "SQD, 1 layer", tone: "q", pts: layers(1) },
        { key: "q2", label: "SQD, 2 layers", tone: "q2", pts: layers(2) },
      ] as Line[],
      time: [
        { key: "tfci", label: "exact (FCI)", tone: "exact", pts: times((r) => r.fci_time_s, study1) },
        { key: "tcc", label: "CCSD", tone: "cc", pts: times((r) => r.ccsd_time_s, study1) },
        { key: "tq", label: "SQD (all)", tone: "q", pts: times((r) => r.wall_time_s, (r) => study1(r) && r.n_reps === 1) },
      ] as Line[],
    };
  }, [rows, groups]);

  if (!data) return null;
  const one = data.errors.find((l) => l.key === "q1")?.pts ?? [];
  const firstPt = one[0];
  const lastPt = one[one.length - 1];
  return (
    <Panel title="Scaling with the number of qubits" aside={`N₂, ${data.sizes.join(", ")} qubits, same settings at every size`}>
      <p className="small muted" style={{ marginBottom: 10, maxWidth: "78ch" }}>
        Every size is checked against its own exact energy. Settings are held fixed (noise 0.02, 3 rounds, 300 samples per batch), so any
        change with size is the method&apos;s, not the settings&apos;. Dots are the mean over seeds, bars are one standard deviation. The sampling
        is simulated; the exact-energy time is the one-off solve (saved after the first run).
      </p>
      <div className="grid grid--2">
        <div>
          <h3 className="scl__h">Error against the exact energy (mHa)</h3>
          <LineChart lines={data.errors} domain={[0.1, 1000]} band axis="mHa, log scale" aria="SQD error against qubit count, with Hartree-Fock and CCSD" />
        </div>
        <div>
          <h3 className="scl__h">Time to solve (seconds)</h3>
          <LineChart lines={data.time} domain={[0.01, 10000]} axis="seconds, log scale" aria="Time against qubit count for exact, CCSD and SQD" />
        </div>
      </div>
      {firstPt && lastPt && firstPt.x !== lastPt.x && (
        <p className="small faint" style={{ marginTop: 8, maxWidth: "78ch" }}>
          With these fixed settings, the one-layer SQD error went from {mha(firstPt.y)} mHa at {firstPt.x} qubits to {mha(lastPt.y)} mHa at{" "}
          {lastPt.x}; chemical accuracy is {CHEM_ACC_MHA} mHa. Every size here can still be solved exactly, which is what makes the check possible.
        </p>
      )}
    </Panel>
  );
}
