import { describe, expect, it } from "vitest";
import { deviceName, mha, parts, pct } from "./format";
import { demoFile } from "./demo";
import { buildReportHtml, reportFileName } from "./trustReport";
import { linear, linearTicks, log, logTicks } from "./scale";
import type { Config } from "../types";

describe("scales", () => {
  it("maps a log domain onto a pixel range", () => {
    const x = log([1, 100], [0, 200]);
    expect(x(1)).toBeCloseTo(0);
    expect(x(10)).toBeCloseTo(100);
    expect(x(100)).toBeCloseTo(200);
  });

  it("clamps values outside a log domain instead of returning NaN", () => {
    const x = log([1, 100], [0, 200]);
    expect(x(0)).toBeCloseTo(0);
    expect(x(-5)).toBeCloseTo(0);
    expect(x(1e6)).toBeCloseTo(200);
  });

  it("maps a linear domain", () => {
    const x = linear([0, 10], [100, 200]);
    expect(x(5)).toBeCloseTo(150);
  });

  it("gives decade ticks inside the domain", () => {
    expect(logTicks([0.5, 200])).toEqual([0.5, 1, 2, 5, 10, 20, 50, 100, 200]);
  });

  it("gives round linear ticks", () => {
    expect(linearTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
  });
});

describe("format", () => {
  it("keeps precision honest across magnitudes", () => {
    expect(mha(0.19)).toBe("0.19");
    expect(mha(3.72)).toBe("3.7");
    expect(mha(84.5)).toBe("84.5");
    expect(mha(360.4)).toBe("360");
    expect(mha(null)).toBe("n/a");
    expect(mha(Number.NaN)).toBe("n/a");
  });

  it("formats shares and device names", () => {
    expect(pct(0.4)).toBe("40%");
    expect(deviceName("FakeSherbrooke")).toBe("Sherbrooke");
  });

  it("describes a configuration in plain words", () => {
    const c = {
      ansatz: "hea_hf",
      reps: 3,
      optimizer: "spsa_clip",
      mitigation: "readout+zne",
      mapping: "parity",
      maxiter: 1500,
      shots: 4096,
    } as Config;
    const p = parts(c);
    expect(p.ansatz).toBe("Hardware-efficient, HF start, 3 layers");
    expect(p.optimizer).toBe("SPSA, clipped");
    expect(p.mitigation).toBe("Readout + ZNE");
    expect(p.budget).toContain("1500 evaluations");
  });
});

// The same literals are asserted in tests/test_demo.py. If one side changes, both fail.
describe("demo file names", () => {
  it("drops everything that does not change an answer", () => {
    expect(demoFile("/api/overview?source=published&n=3")).toBe("_api_overview.json");
  });
  it("keeps the parameters that do, in a stable order", () => {
    expect(demoFile("/api/rank?molecule=H2&backend=noisy&bond_length=0.735&w_acc=70&source=published")).toBe(
      "_api_rank_backend=noisy_bond_length=0.735_molecule=H2.json",
    );
    expect(demoFile("/api/rank?molecule=H2&backend=ideal&bond_length=1.0")).toBe(
      "_api_rank_backend=ideal_bond_length=1_molecule=H2.json",
    );
    expect(demoFile("/api/rank?molecule=LiH&backend=noisy")).toBe("_api_rank_backend=noisy_molecule=LiH.json");
    expect(
      demoFile("/api/replicated/recommendation?molecule=H2&backend=noisy&device=FakeFez&reliability=0.5"),
    ).toBe("_api_replicated_recommendation_backend=noisy_device=FakeFez_molecule=H2.json");
  });
  it("names the SQD views", () => {
    expect(demoFile("/api/sqd?source=published&n=2")).toBe("_api_sqd.json");
    expect(demoFile("/api/sqd/options")).toBe("_api_sqd_options.json");
    expect(demoFile("/api/structures")).toBe("_api_structures.json");
  });
  it("names the trust reports by what picks them out", () => {
    expect(demoFile("/api/trust/index?source=published")).toBe("_api_trust_index.json");
    expect(demoFile("/api/trust?kind=vqe&molecule=H2&backend=noisy&device=FakeFez&source=published&n=2")).toBe(
      "_api_trust_backend=noisy_device=FakeFez_kind=vqe_molecule=H2.json",
    );
    expect(demoFile("/api/trust?kind=vqe&molecule=LiH&backend=ideal&device=")).toBe(
      "_api_trust_backend=ideal_kind=vqe_molecule=LiH.json",
    );
  });
  it("maps a run to its own file", () => {
    expect(demoFile("/api/runs/abc123?source=published")).toBe("runs/abc123.json");
  });
});

describe("trust report file", () => {
  const report = {
    kind: "vqe" as const,
    subject: { molecule: "H2", backend: "noisy", device: "fake_sherbrooke", bond_length: null, label: "SPSA <fast>" },
    verdict: "not_yet" as const,
    verdict_text: "Not yet trusted",
    sentence: "1 of 10 seeds reached chemical accuracy.",
    evidence: [
      { id: "seeds", title: "Seeds", status: "pass" as const, detail: "10 seeds.", route: null },
      { id: "acc", title: "Accuracy", status: "fail" as const, detail: "1 of 10 within 1.6 mHa.", route: null },
    ],
    rules: ["Fewer than 3 seeds is not enough evidence."],
  };

  it("builds a self-contained document with the sections of a short paper", () => {
    const html = buildReportHtml(report, "H2, VQE", new Date("2026-10-31T00:00:00Z"));
    for (const h of ["Summary", "1. Method", "2. Evidence", "3. How the verdict is decided", "4. Limitations"]) {
      expect(html).toContain(h);
    }
    expect(html).toContain("Not yet trusted");
    expect(html).toContain("2026-10-31");
    expect(html).toContain("1 passed, 1 failed");
    expect(html).not.toContain("<script");
  });

  it("escapes text taken from the report", () => {
    expect(buildReportHtml(report, "H2, VQE", new Date())).toContain("SPSA &lt;fast&gt;");
  });

  it("names the file after the subject", () => {
    expect(reportFileName("H₂, VQE, noisy")).toBe("miraichem-trust-report-h-vqe-noisy.html");
  });
});
