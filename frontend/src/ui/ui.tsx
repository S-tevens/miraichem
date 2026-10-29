import { useState, type ReactNode } from "react";
import { DotArt, type DotKind } from "../components/DotArt";

export function PageHead({
  title,
  sub,
  tools,
}: {
  title: string;
  sub?: ReactNode;
  tools?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div>
        <h1 className="page-head__title">{title}</h1>
        {sub && <p className="page-head__sub">{sub}</p>}
      </div>
      {tools && <div className="page-head__tools">{tools}</div>}
    </header>
  );
}

export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || aside) && (
        <div className="panel__head">
          {title && <h2 className="panel__title">{title}</h2>}
          {aside && <div className="muted small">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  unit,
  note,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  note?: ReactNode;
}) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value num">
        {value}
        {unit && <small> {unit}</small>}
      </span>
      {note && <span className="stat__note">{note}</span>}
    </div>
  );
}

export function Seg<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { id: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={o.id === value} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}

export function Select({
  label,
  value,
  onChange,
  options,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { id: string; label: string }[];
  hint?: ReactNode;
}) {
  return (
    <Field label={label} hint={hint}>
      <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function Progress({
  value,
  total,
  tone,
}: {
  value: number;
  total: number;
  tone?: "ok" | "bad";
}) {
  const pct = total > 0 ? Math.min(100, (100 * value) / total) : 0;
  return (
    <div
      className="bar"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={value}
    >
      <div className={`bar__fill ${tone ? `bar__fill--${tone}` : ""}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Empty({
  title,
  children,
  art = "grid",
  action,
}: {
  title: string;
  children?: ReactNode;
  art?: DotKind;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <DotArt kind={art} className="empty__art" />
      <strong style={{ fontWeight: 400, color: "var(--fg)" }}>{title}</strong>
      {children && <p className="small" style={{ maxWidth: "46ch" }}>{children}</p>}
      {action}
    </div>
  );
}

export function Notice({
  tone,
  children,
}: {
  tone?: "warn" | "bad";
  children: ReactNode;
}) {
  return <div className={`notice ${tone ? `notice--${tone}` : ""}`}>{children}</div>;
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="empty" role="status">
      <div className="spinner" />
      <span className="small">{label}…</span>
    </div>
  );
}

export function CopyButton({ text, label = "Copy command" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="btn btn--ghost btn--small"
      onClick={() => {
        navigator.clipboard
          ?.writeText(text)
          .then(() => {
            setDone(true);
            setTimeout(() => setDone(false), 1600);
          })
          .catch(() => setDone(false));
      }}
    >
      {done ? "Copied" : label}
    </button>
  );
}

// ------------------------------------------------------------ icons (stroke, 18px)

const P = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const ICONS: Record<string, ReactNode> = {
  overview: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" {...P} />
      <rect x="14" y="3" width="7" height="5" rx="1.5" {...P} />
      <rect x="14" y="12" width="7" height="9" rx="1.5" {...P} />
      <rect x="3" y="14" width="7" height="7" rx="1.5" {...P} />
    </>
  ),
  leaderboard: <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" {...P} />,
  reliability: (
    <>
      <circle cx="6" cy="14" r="1.6" {...P} />
      <circle cx="11" cy="9" r="1.6" {...P} />
      <circle cx="15" cy="15" r="1.6" {...P} />
      <circle cx="19" cy="7" r="1.6" {...P} />
      <path d="M3 20h18" {...P} />
    </>
  ),
  devices: (
    <>
      <circle cx="6" cy="7" r="2" {...P} />
      <circle cx="12" cy="7" r="3" {...P} />
      <circle cx="18" cy="7" r="2" {...P} />
      <circle cx="6" cy="16" r="3" {...P} />
      <circle cx="12" cy="16" r="2" {...P} />
      <circle cx="18" cy="16" r="3" {...P} />
    </>
  ),
  curves: <path d="M3 5c3 0 4 14 8 14s4-9 10-9" {...P} />,
  hardware: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" {...P} />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" {...P} />
    </>
  ),
  predictor: (
    <>
      <path d="M3 17l5-6 4 4 8-9" {...P} />
      <path d="M16 6h4v4" {...P} />
    </>
  ),
  sqd: (
    <>
      <circle cx="12" cy="12" r="2" {...P} />
      <circle cx="5" cy="6" r="1.4" {...P} />
      <circle cx="19" cy="6" r="1.4" {...P} />
      <circle cx="5" cy="18" r="1.4" {...P} />
      <circle cx="19" cy="18" r="1.4" {...P} />
      <path d="M6.2 7l4.4 3.6M17.8 7l-4.4 3.6M6.2 17l4.4-3.6M17.8 17l-4.4-3.6" {...P} />
    </>
  ),
  "new-run": <path d="M7 4l12 8-12 8V4z" {...P} />,
  sweeps: (
    <>
      <path d="M4 6h16M4 12h10M4 18h6" {...P} />
      <path d="M17 15l3 3-3 3" {...P} />
    </>
  ),
  jobs: (
    <>
      <circle cx="12" cy="12" r="9" {...P} />
      <path d="M12 7v5l3 2" {...P} />
    </>
  ),
  about: (
    <>
      <circle cx="12" cy="12" r="9" {...P} />
      <path d="M12 11v5M12 8h.01" {...P} />
    </>
  ),
};

export function Icon({ name, className = "nav__icon" }: { name: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}
