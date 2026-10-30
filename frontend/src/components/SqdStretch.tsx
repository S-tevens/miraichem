import { useMemo } from "react";
import { linear, log, logTicks } from "../lib/scale";
import { mha } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { Panel } from "../ui/ui";
import { CHEM_ACC_MHA, type SqdGroup } from "../types";
import "./charts.css";
import "./SqdScaling.css";

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const molName = (n: string) => {
  const [head, ...rest] = n.split("_");
  const f = head.replace(/\d/g, (d) => SUB[Number(d)]);
  return rest.length ? `${f} (${rest.join(" ")})` : f;
};

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

/** The settings of the stretched-bond study; they match the equilibrium seed study. */
const isStudy = (g: SqdGroup) =>
  g.noise_model === null &&
  g.readout_noise === 0.02 &&
  g.iterations === 3 &&
  g.samples_per_batch === 300 &&
  g.num_batches === 3 &&
  g.shots === 20000;

function BondChart({ lines, domain }: { lines: Line[]; domain: [number, number] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const xs = [...new Set(lines.flatMap((l) => l.pts.map((p) => p.x)))].sort((a, b) => a - b);
  const left = 54;
  const right = 130;
  const top = 14;
  const height = 280;
  const bottom = height - 42;
  const x = linear([xs[0] - 0.05, xs[xs.length - 1] + 0.05], [left, Math.max(left + 160, width - right)]);
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
      <svg width={width} height={height} role="img" aria-label="Error against bond length for Hartree-Fock, CCSD and SQD">
        <rect x={left} y={y(CHEM_ACC_MHA)} width={Math.max(0, x(xs[xs.length - 1] + 0.05) - left)} height={Math.max(0, y(domain[0]) - y(CHEM_ACC_MHA))} className="chart__band" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={x(xs[xs.length - 1] + 0.05)} y1={y(t)} y2={y(t)} className="chart__grid" />
            <text x={left - 8} y={y(t) + 4} textAnchor="end" className="chart__axis num">{t >= 1000 ? `${t / 1000}k` : t}</text>
          </g>
        ))}
        {xs.map((b) => (
          <text key={b} x={x(b)} y={bottom + 18} textAnchor="middle" className="chart__axis num">{b}</text>
        ))}
        <text x={x(xs[Math.floor(xs.length / 2)])} y={height - 6} textAnchor="middle" className="chart__axis">N–N bond length (Å)</text>
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
                    <title>{`${l.label}, ${p.x} Å: ${p.note}`}</title>
                  </circle>
                </g>
              ))}
              <text x={x(last.x) + 12} y={labelY + 4} className="scl__label">{l.label}</text>
            </g>
          );
        })}
        <text x={left} y={top - 2} className="chart__axis">error against the exact energy, mHa (log scale)</text>
      </svg>
    </div>
  );
}

/**
 * Error against bond length for one molecule. Classical coupled cluster is accurate near the
 * equilibrium distance and falls apart as the bond stretches (strong correlation); this shows what
 * SQD does over the same range, checked against the exact energy at every point.
 */
export function SqdStretch({ groups, molecule }: { groups: SqdGroup[]; molecule: string }) {
  const data = useMemo(() => {
    const study = groups.filter((g) => g.molecule === molecule && isStudy(g));
    const bonds = [...new Set(study.map((g) => g.bond_length))].sort((a, b) => a - b);
    if (bonds.length < 3) return null;
    const at = (b: number, reps: number) => study.find((g) => g.bond_length === b && g.n_reps === reps);
    const base = (b: number) => at(b, 1) ?? at(b, 2);
    const hf: Pt[] = [];
    const cc: Pt[] = [];
    const rows = bonds.map((b) => {
      const g = base(b)!;
      if (g.hf_error_mha) hf.push({ x: b, y: g.hf_error_mha, note: `${mha(g.hf_error_mha)} mHa` });
      if (g.ccsd_error_mha) cc.push({ x: b, y: g.ccsd_error_mha, note: `${mha(g.ccsd_error_mha)} mHa` });
      return { b, g, one: at(b, 1), two: at(b, 2) };
    });
    const layer = (n: 1 | 2): Pt[] =>
      rows.flatMap((r) => {
        const s = n === 1 ? r.one : r.two;
        return s
          ? [
              {
                x: r.b,
                y: Math.max(s.mean_error_mha, 0.05),
                lo: Math.max(0.05, s.mean_error_mha - s.std_error_mha),
                hi: s.mean_error_mha + s.std_error_mha,
                note: `${mha(s.mean_error_mha)} ± ${mha(s.std_error_mha)} mHa over ${s.n_seeds} seeds`,
              },
            ]
          : [];
      });
    const pairs = rows.filter((r) => r.one && r.g.ccsd_error_mha !== null);
    const closer = pairs.filter((r) => r.one!.mean_error_mha < (r.g.ccsd_error_mha ?? Infinity));
    const worstCc = [...rows].sort((a, b) => (b.g.ccsd_error_mha ?? 0) - (a.g.ccsd_error_mha ?? 0))[0];
    const fractions = rows.flatMap((r) => (r.one ? [r.one.mean_subspace_fraction] : []));
    return {
      bonds,
      rows,
      lines: [
        { key: "hf", label: "Hartree-Fock", tone: "hf", pts: hf },
        { key: "cc", label: "CCSD", tone: "cc", pts: cc },
        { key: "q1", label: "SQD, 1 layer", tone: "q", pts: layer(1) },
        { key: "q2", label: "SQD, 2 layers", tone: "q2", pts: layer(2) },
      ] as Line[],
      closer: closer.length,
      of: pairs.length,
      worstCc,
      fMin: Math.min(...fractions),
      fMax: Math.max(...fractions),
    };
  }, [groups, molecule]);

  if (!data) return null;
  const { worstCc } = data;
  return (
    <Panel title="Stretched bonds: where classical methods struggle" aside={`${molName(molecule)}, every point checked against the exact energy`}>
      <p className="small muted" style={{ marginBottom: 10, maxWidth: "78ch" }}>
        Near the equilibrium distance the bond is easy for classical methods. As the two atoms move apart the electrons become strongly
        correlated, and coupled cluster (CCSD) loses accuracy. This runs the same SQD setting at every distance (noise 0.02, 3 rounds, 300
        samples per batch). Dots are the mean over seeds and bars are one standard deviation. The sampling is simulated. Errors are shown as
        distances from the exact energy, so a CCSD energy that falls below the exact one still shows as a positive error.
      </p>
      <BondChart lines={data.lines} domain={[0.1, 1000]} />
      <div className="tablewrap" style={{ marginTop: 14 }}>
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Bond (Å)</th>
              <th scope="col" className="right">Hartree-Fock</th>
              <th scope="col" className="right">CCSD</th>
              <th scope="col" className="right">SQD, 1 layer</th>
              <th scope="col" className="right">SQD, 2 layers</th>
              <th scope="col" className="right">Space solved</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => (
              <tr key={r.b}>
                <td className="num">{r.b}</td>
                <td className="right num">{mha(r.g.hf_error_mha)}</td>
                <td className="right num">{mha(r.g.ccsd_error_mha)}</td>
                <td className="right num">{r.one ? `${mha(r.one.mean_error_mha)} ± ${mha(r.one.std_error_mha)}` : "n/a"}</td>
                <td className="right num">{r.two ? `${mha(r.two.mean_error_mha)} ± ${mha(r.two.std_error_mha)}` : "n/a"}</td>
                <td className="right num">{r.one ? `${(100 * r.one.mean_subspace_fraction).toFixed(0)}%` : "n/a"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="small muted" style={{ margin: "12px 0 0", paddingLeft: 18, display: "grid", gap: 4, maxWidth: "78ch" }}>
        <li>
          CCSD error grows to {mha(worstCc.g.ccsd_error_mha)} mHa at {worstCc.b} Å, against {mha(data.rows[0].g.ccsd_error_mha)} mHa at {data.bonds[0]} Å.
        </li>
        <li>
          SQD (1 layer) is closer to the exact energy than CCSD at {data.closer} of {data.of} bond lengths.
        </li>
        <li>
          SQD solves between {(100 * data.fMin).toFixed(0)}% and {(100 * data.fMax).toFixed(0)}% of the full space here. When that share is large, SQD is
          close to exact diagonalisation, so these small cases show the method working, not how it behaves on a molecule too big to solve exactly.
        </li>
      </ul>
    </Panel>
  );
}
