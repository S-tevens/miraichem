/** Tiny scale helpers (no d3): enough for the log-axis charts on this page. */

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** Log10 scale; values below the domain floor are clamped so zeros never break the axis. */
export function log(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const f = ((v: number) => {
    const c = Math.min(Math.max(v, d0), d1);
    return r0 + ((Math.log10(c) - l0) / (l1 - l0)) * (r1 - r0);
  }) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** Nice decade ticks (1, 10, 100, ...) with 2 and 5 sub-ticks when the range is narrow. */
export function logTicks(domain: [number, number]): number[] {
  const [lo, hi] = domain;
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

export function linearTicks(lo: number, hi: number, count = 5): number[] {
  const step = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step));
  const nice = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step) ?? step;
  const out: number[] = [];
  for (let v = Math.ceil(lo / nice) * nice; v <= hi + 1e-9; v += nice) out.push(+v.toFixed(10));
  return out;
}

export function tickLabel(v: number): string {
  if (v >= 1000) return `${v / 1000}k`;
  if (v < 1) return `${v}`;
  return `${v}`;
}
