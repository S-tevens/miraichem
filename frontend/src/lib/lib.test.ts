import { describe, expect, it } from "vitest";
import { deviceName, mha, parts, pct } from "./format";
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
