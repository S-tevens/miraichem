import type { TrustReport, TrustStatus } from "../types";

const STATUS_WORD: Record<TrustStatus, string> = { pass: "Passes", warn: "Warning", fail: "Fails", none: "No data" };
const COUNT_WORD: Record<TrustStatus, string> = { pass: "passed", warn: "with a warning", fail: "failed", none: "with no data" };

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** A file name for the report, built from the subject label. */
export function reportFileName(label: string): string {
  const slug = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `miraichem-trust-report-${slug || "report"}.html`;
}

/**
 * The report as one self-contained HTML document laid out like a short paper: summary, method, results,
 * verdict rules and limitations. It has no scripts and no external files, and wraps at any width.
 */
export function buildReportHtml(report: TrustReport, label: string, generated: Date): string {
  const when = generated.toISOString().slice(0, 10);
  const counts = (["pass", "warn", "fail", "none"] as TrustStatus[])
    .map((s) => [s, report.evidence.filter((e) => e.status === s).length] as const)
    .filter(([, n]) => n > 0)
    .map(([s, n]) => `${n} ${COUNT_WORD[s]}`)
    .join(", ");

  const rows = report.evidence
    .map(
      (e, i) =>
        `<tr><td class="n">${i + 1}</td><td>${esc(e.title)}</td><td class="s s-${e.status}">${STATUS_WORD[e.status]}</td><td>${esc(e.detail)}</td></tr>`,
    )
    .join("\n");
  const rules = report.rules.map((r) => `<li>${esc(r)}</li>`).join("\n");
  const method =
    report.kind === "sqd"
      ? "Sample-based quantum diagonalization (SQD) results for this molecule were compared with the exact energy and with classical coupled cluster (CCSD). Where available, control experiments (equal-size random and classical subspaces, a swapped-in sampler), a stretched-bond study and real-device runs were also checked."
      : "A variational quantum eigensolver (VQE) setup was run for several random seeds and compared with the exact energy. The recommended setup was then checked for seed reliability, agreement across devices, the error forecast and, where a matching run exists, real-hardware evidence.";

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trust report: ${esc(label)}</title>
<style>
  body { font: 16px/1.6 Georgia, "Times New Roman", serif; color: #111; background: #fff; margin: 0; }
  main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; overflow-wrap: anywhere; }
  h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .3rem; }
  h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
  .meta { color: #444; font-size: .9rem; margin: 0 0 1.5rem; }
  .verdict { border: 1px solid #999; padding: .8rem 1rem; margin: 1rem 0; }
  .verdict strong { font-size: 1.1rem; }
  table { width: 100%; border-collapse: collapse; font-size: .92rem; table-layout: fixed; }
  th, td { border: 1px solid #bbb; padding: .4rem .5rem; text-align: left; vertical-align: top; }
  th { background: #f1f1f1; }
  td.n { width: 2rem; text-align: center; }
  th:nth-child(1) { width: 2rem; }
  th:nth-child(2) { width: 24%; }
  th:nth-child(3), td.s { width: 5.5rem; }
  .s-pass { color: #1b6b3a; } .s-warn { color: #8a5a00; } .s-fail { color: #a11; } .s-none { color: #555; }
  ol, ul { padding-left: 1.3rem; }
  .foot { color: #555; font-size: .85rem; margin-top: 2.5rem; border-top: 1px solid #ccc; padding-top: .6rem; }
  @media print { main { padding: 0; } tr { break-inside: avoid; } }
</style>
</head>
<body>
<main>
<h1>Trust report: ${esc(label)}</h1>
<p class="meta">MiraiChem &middot; Team Mirai &middot; generated ${when}</p>

<h2>Summary</h2>
<div class="verdict"><strong>${esc(report.verdict_text)}</strong><br>${esc(report.sentence)}</div>
${report.subject.label ? `<p>Recommended setup: ${esc(report.subject.label)}.</p>` : ""}
<p>${report.evidence.length} checks were made: ${esc(counts || "none")}.</p>

<h2>1. Method</h2>
<p>${esc(method)} Every number comes from saved result files, each stored with its configuration, seed, library versions and code version.</p>

<h2>2. Evidence</h2>
<table>
<thead><tr><th>#</th><th>Check</th><th>Outcome</th><th>Finding</th></tr></thead>
<tbody>
${rows}
</tbody>
</table>

<h2>3. How the verdict is decided</h2>
<ol>
${rules}
</ol>
<p>A check with no data is shown as such and does not count against a setup.</p>

<h2>4. Limitations</h2>
<ul>
<li>A trusted verdict describes behaviour on small molecules that can be checked against an exact answer. It is not a claim that the method helps on molecules too large to solve classically.</li>
<li>No result here is a quantum advantage, and no real-device result is claimed to reach chemical accuracy unless the evidence above says so.</li>
<li>Simulators leave out drift and crosstalk. Real-device results are single runs.</li>
</ul>

<p class="foot">Produced by MiraiChem (github.com/S-tevens/miraichem). Chemical accuracy means an energy error below 1.6 mHa.</p>
</main>
</body>
</html>
`;
}

/** Hand the report to the browser as a file download. */
export function downloadReport(report: TrustReport, label: string): void {
  const html = buildReportHtml(report, label, new Date());
  const url = URL.createObjectURL(new Blob([html], { type: "text/html;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = reportFileName(label);
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
