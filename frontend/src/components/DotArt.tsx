/**
 * Small dot-matrix illustrations in the reference's style. Each pattern is generated, not drawn,
 * so it stays crisp at any size and needs no image files.
 */
export type DotKind = "arc" | "grid" | "stagger" | "spiral" | "scatter" | "blocks" | "rings";

interface Dot {
  x: number;
  y: number;
  r: number;
  o: number;
}

/** Small deterministic pseudo-random generator so the art is identical on every render. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function dots(kind: DotKind, w = 240, h = 120): Dot[] {
  const out: Dot[] = [];
  const rand = rng(kind.length * 977 + 13);
  switch (kind) {
    case "arc": {
      for (let ring = 0; ring < 7; ring++) {
        const R = 40 + ring * 15;
        const n = 10 + ring * 4;
        for (let i = 0; i < n; i++) {
          const a = Math.PI * (0.08 + (0.84 * i) / (n - 1));
          out.push({ x: w / 2 + R * Math.cos(a) * 1.5, y: h + 8 - R * Math.sin(a) * 0.95, r: 1.4 + ring * 0.16, o: 0.35 + ring * 0.08 });
        }
      }
      break;
    }
    case "grid": {
      for (let j = 0; j < 8; j++)
        for (let i = 0; i < 16; i++) {
          const lit = rand() > 0.82;
          out.push({ x: 14 + i * 14.5, y: 12 + j * 14.5, r: lit ? 3 : 1.7, o: lit ? 1 : 0.38 });
        }
      break;
    }
    case "stagger": {
      for (let j = 0; j < 9; j++)
        for (let i = 0; i < 16; i++) {
          const wave = Math.sin(i * 0.5 + j * 0.7);
          out.push({ x: 12 + i * 14.5 + (j % 2 ? 7 : 0), y: 10 + j * 13, r: 1.2 + Math.max(0, wave) * 2.4, o: 0.3 + Math.max(0, wave) * 0.6 });
        }
      break;
    }
    case "spiral": {
      for (let i = 0; i < 150; i++) {
        const a = i * 0.34;
        const R = 3 + i * 0.62;
        out.push({ x: w / 2 + R * Math.cos(a) * 1.35, y: h / 2 + R * Math.sin(a) * 0.78, r: 1.1 + (i / 150) * 1.9, o: 0.25 + (i / 150) * 0.7 });
      }
      break;
    }
    case "scatter": {
      for (let i = 0; i < 120; i++) {
        const cluster = i % 3;
        const cx = [60, 120, 185][cluster];
        const cy = [70, 40, 78][cluster];
        out.push({ x: cx + (rand() - 0.5) * 70, y: cy + (rand() - 0.5) * 60, r: 1.2 + rand() * 2.4, o: 0.3 + rand() * 0.7 });
      }
      break;
    }
    case "blocks": {
      for (let j = 0; j < 8; j++)
        for (let i = 0; i < 16; i++) {
          const inBlock = (i > 1 && i < 6 && j > 0 && j < 5) || (i > 9 && i < 14 && j > 2 && j < 7);
          out.push({ x: 14 + i * 14.5, y: 12 + j * 14.5, r: inBlock ? 3 : 1.4, o: inBlock ? 0.95 : 0.28 });
        }
      break;
    }
    case "rings": {
      for (let ring = 1; ring <= 5; ring++) {
        const n = ring * 9;
        for (let i = 0; i < n; i++) {
          const a = (i / n) * Math.PI * 2;
          out.push({ x: w / 2 + ring * 19 * Math.cos(a) * 1.4, y: h / 2 + ring * 11 * Math.sin(a), r: 1.3 + (6 - ring) * 0.22, o: 0.3 + (6 - ring) * 0.1 });
        }
      }
      break;
    }
  }
  return out;
}

export function DotArt({ kind, className }: { kind: DotKind; className?: string }) {
  const W = 240;
  const H = 120;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} role="presentation" aria-hidden="true" preserveAspectRatio="xMidYMid slice">
      {dots(kind, W, H).map((d, i) => (
        <circle key={i} cx={d.x} cy={d.y} r={d.r} fill="#a9d6d1" opacity={d.o} />
      ))}
    </svg>
  );
}
