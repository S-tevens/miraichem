import { useMemo, useState } from "react";
import { qs, useQuery } from "../api";
import { useJobs, useSource } from "../app/state";
import { linear, linearTicks, log, logTicks } from "../lib/scale";
import { mha, moleculeName } from "../lib/format";
import { useWidth } from "../lib/hooks";
import { Empty, Loading, Notice, PageHead, Panel, Select } from "../ui/ui";
import { CHEM_ACC_MHA, type Curve } from "../types";
import "../components/charts.css";
import "./Curves.css";

function curveName(c: Curve): string {
  return c.label
    .replace(/^uccsd\(reps=1\)/, "UCCSD")
    .replace(/^hea\(reps=(\d)\)/, "Hardware-efficient, $1 layers")
    .replace(/\/lbfgsb/, " · L-BFGS-B")
    .replace(/\/spsa_clip/, " · SPSA, clipped")
    .replace(/\/spsa/, " · SPSA")
    .replace(/\/cobyla/, " · COBYLA")
    .replace(/, parity/, "")
    .replace(/, noisy/, ", noisy simulator")
    .replace(/, ideal/, ", ideal simulator");
}

const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");

export function CurvesPage() {
  const { source } = useSource();
  const { finishedCount } = useJobs();
  const q = useQuery<{ curves: Curve[] }>(`/api/curves${qs({ source, n: finishedCount })}`);
  const curves = useMemo(() => {
    const seen = new Set<string>();
    return (q.data?.curves ?? [])
      .filter((c) => {
        const k = `${c.molecule}|${curveName(c)}`;
        if (seen.has(k)) return false; // the same setup saved twice: show it once
        seen.add(k);
        return true;
      })
      .sort((a, b) => Number(a.backend === "noisy") - Number(b.backend === "noisy"));
  }, [q.data]);
  const [idx, setIdx] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [ref, width] = useWidth<HTMLDivElement>();

  if (q.loading && !q.data) return <div className="page"><Loading /></div>;
  if (q.error) return <div className="page"><Notice tone="bad">{q.error}</Notice></div>;
  if (!curves.length) {
    return (
      <div className="page">
        <PageHead title="Dissociation curves" />
        <Panel>
          <Empty title="No curves saved in this source" art="spiral">
            Create one from the command line with <code>miraichem curve --molecule configs/molecules/h2.yaml</code>.
          </Empty>
        </Panel>
      </div>
    );
  }

  const defaultIdx = Math.max(0, curves.findIndex((c) => c.backend === "noisy" && c.label.startsWith("uccsd")));
  const c = curves[Math.min(idx ?? defaultIdx, curves.length - 1)];

  const left = 62;
  const right = 18;
  const w1 = Math.max(280, width);
  const xs = linear([Math.min(...c.bond_lengths), Math.max(...c.bond_lengths)], [left, w1 - right]);

  const h1 = 280;
  const vqe = c.e_vqe.map((v) => v ?? NaN);
  const all = [...c.e_hf, ...c.e_exact, ...vqe.filter(Number.isFinite)];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.08;
  const ye = linear([lo - pad, hi + pad], [h1 - 34, 14]);
  const eTicks = linearTicks(lo, hi, 4);
  const line = (vals: number[]) =>
    path(c.bond_lengths.map((b, i) => [xs(b), ye(vals[i])] as [number, number]).filter(([, y]) => Number.isFinite(y)));

  const h2 = 210;
  const errs = c.error_mha.map((v) => v ?? NaN);
  const eMax = Math.max(...c.hf_error_mha, ...errs.filter(Number.isFinite), 10);
  const dom: [number, number] = [0.1, 10 ** Math.ceil(Math.log10(eMax * 1.2))];
  const yl = log(dom, [h2 - 32, 12]);
  const lt = logTicks(dom).filter((t) => [0.1, 1, 10, 100, 1000].includes(t));

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = 0;
    let bd = Infinity;
    c.bond_lengths.forEach((b, i) => {
      const d = Math.abs(xs(b) - px);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    setHover(best);
  };

  return (
    <div className="page">
      <PageHead
        title="Dissociation curves"
        sub="Energy of the molecule as its two atoms move apart. Near the equilibrium distance Hartree-Fock is decent; stretched, it fails because the electrons become strongly correlated, which is where a quantum method is meant to help."
        tools={
          <Select
            label="Curve"
            value={String(Math.min(idx ?? defaultIdx, curves.length - 1))}
            onChange={(v) => {
              setIdx(Number(v));
              setHover(null);
            }}
            options={curves.map((q2, k) => ({ id: String(k), label: `${moleculeName(q2.molecule)}: ${curveName(q2)}` }))}
          />
        }
      />

      <Panel title="Energy" aside={`${moleculeName(c.molecule)} · ${c.bond_lengths.length} bond lengths`}>
        <div ref={ref} className="chart curves">
          <svg width={w1} height={h1} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Total energy against bond length for exact, Hartree-Fock and VQE">
            {eTicks.map((t) => (
              <g key={t}>
                <line x1={left} x2={w1 - right} y1={ye(t)} y2={ye(t)} className="chart__grid" />
                <text x={left - 10} y={ye(t) + 4} textAnchor="end" className="chart__axis num">{t.toFixed(2)}</text>
              </g>
            ))}
            <path d={line(c.e_hf)} className="cv__line cv__line--hf" />
            <path d={line(c.e_exact)} className="cv__line cv__line--exact" />
            <path d={line(vqe)} className="cv__line cv__line--vqe" />
            {c.bond_lengths.map((b, i) => (
              <circle key={b} cx={xs(b)} cy={Number.isFinite(vqe[i]) ? ye(vqe[i]) : -10} r={hover === i ? 6 : 3.5} className="cv__pt" />
            ))}
            {hover !== null && <line x1={xs(c.bond_lengths[hover])} x2={xs(c.bond_lengths[hover])} y1={10} y2={h1 - 34} className="chart__cross" />}
            <text x={14} y={h1 / 2} transform={`rotate(-90 14 ${h1 / 2})`} textAnchor="middle" className="chart__axis">
              total energy (Hartree)
            </text>
          </svg>
          <div className="chart__key small">
            <span><i className="sw sw--exact" /> exact</span>
            <span><i className="sw sw--hf" /> Hartree-Fock</span>
            <span><i className="sw sw--vqe" /> VQE</span>
          </div>
        </div>
      </Panel>

      <Panel title="Error" aside="log scale; shaded band is chemical accuracy">
        <div className="chart curves">
          <svg width={w1} height={h2} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Error against bond length">
            <rect x={left} y={yl(CHEM_ACC_MHA)} width={w1 - right - left} height={Math.max(0, yl(dom[0]) - yl(CHEM_ACC_MHA))} className="chart__band" />
            {lt.map((t) => (
              <g key={t}>
                <line x1={left} x2={w1 - right} y1={yl(t)} y2={yl(t)} className="chart__grid" />
                <text x={left - 10} y={yl(t) + 4} textAnchor="end" className="chart__axis num">{t}</text>
              </g>
            ))}
            <path d={path(c.bond_lengths.map((b, i) => [xs(b), yl(c.hf_error_mha[i])] as [number, number]))} className="cv__line cv__line--hf" />
            <path
              d={path(
                c.bond_lengths
                  .map((b, i) => [xs(b), yl(Math.max(errs[i], dom[0]))] as [number, number])
                  .filter((_, i) => Number.isFinite(errs[i])),
              )}
              className="cv__line cv__line--vqe"
            />
            {hover !== null && <line x1={xs(c.bond_lengths[hover])} x2={xs(c.bond_lengths[hover])} y1={10} y2={h2 - 32} className="chart__cross" />}
            {linearTicks(c.bond_lengths[0], c.bond_lengths[c.bond_lengths.length - 1], 5).map((t) => (
              <text key={t} x={xs(t)} y={h2 - 10} textAnchor="middle" className="chart__axis num">{t}</text>
            ))}
            <text x={w1 - right} y={h2 - 0} textAnchor="end" className="chart__axis">bond length (Å)</text>
            <text x={14} y={h2 / 2} transform={`rotate(-90 14 ${h2 / 2})`} textAnchor="middle" className="chart__axis">error (mHa)</text>
          </svg>
          {hover !== null && (
            <div className="curves__tip small num" style={{ left: Math.min(xs(c.bond_lengths[hover]), w1 - 210) }}>
              <strong>{c.bond_lengths[hover]} Å</strong>
              <br />
              exact {c.e_exact[hover].toFixed(4)} Ha
              <br />
              HF error {mha(c.hf_error_mha[hover])} mHa · VQE error {mha(c.error_mha[hover])} mHa
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
