"""Dissociation curves: total energy vs bond length for HF, VQE and the exact answer.

Pulling a molecule apart is the classic test of a quantum chemistry method. Near equilibrium
Hartree-Fock (one electron configuration) is decent; as the bond stretches it fails badly because
the electrons become strongly correlated. A good VQE should follow the exact curve everywhere.
All energies are total energies in Hartree; bond lengths are in Angstrom.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.benchmark.ranking import ERROR_FLOOR_MHA
from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR
from miraichem.benchmark.sweep import run_sweep
from miraichem.config import RunConfig, SweepConfig
from miraichem.vqe.result import RunResult

# Sensible default scans (Angstrom). H2: 0.3-2.5 as in the project brief, with the equilibrium.
DEFAULT_BOND_LENGTHS: dict[str, list[float]] = {
    "h2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.735, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5],
    "lih": [1.0, 1.2, 1.4, 1.595, 1.8, 2.0, 2.4, 2.8, 3.2],
}


def default_bond_lengths(molecule: str) -> list[float]:
    """Default scan for a molecule name (falls back to 12 points from 0.5 to 2.5 A)."""
    return DEFAULT_BOND_LENGTHS.get(
        molecule.lower(), [round(x, 3) for x in np.linspace(0.5, 2.5, 12)]
    )


@dataclass
class DissociationCurve:
    """One curve: reference energies and VQE energies at each bond length (Hartree)."""

    molecule: str
    label: str  # human-readable configuration description
    config_hash: str  # hash of the base configuration (bond length excluded from meaning)
    bond_lengths: list[float]
    e_hf: list[float]
    e_exact: list[float]
    e_vqe: list[float | None]  # None where the run failed
    backend: str = "ideal"
    failed: list[float] = field(default_factory=list)  # bond lengths whose run failed

    @property
    def error_mha(self) -> list[float | None]:
        pairs = zip(self.e_vqe, self.e_exact, strict=True)
        return [None if v is None else abs(v - ex) * 1000 for v, ex in pairs]

    @property
    def hf_error_mha(self) -> list[float]:
        return [abs(h - ex) * 1000 for h, ex in zip(self.e_hf, self.e_exact, strict=True)]

    @property
    def max_vqe_error_mha(self) -> float | None:
        errs = [e for e in self.error_mha if e is not None]
        return max(errs) if errs else None

    @property
    def fraction_chemically_accurate(self) -> float:
        errs = [e for e in self.error_mha if e is not None]
        if not errs:
            return 0.0
        return sum(e < CHEMICAL_ACCURACY_HA * 1000 for e in errs) / len(errs)

    def to_json(self) -> str:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return json.dumps(data, indent=2)


def sweep_for_config(base: RunConfig, bond_lengths: list[float]) -> SweepConfig:
    """A one-configuration sweep over bond lengths (everything but the bond length fixed)."""
    return SweepConfig(
        molecule=base.molecule,
        bond_lengths=bond_lengths,
        mappings=[base.mapping],
        ansatzes=[base.ansatz],
        ansatz_reps=[base.ansatz_reps],
        optimizers=[base.optimizer],
        maxiter=base.maxiter,
        backends=[base.backend],
        fake_backend_name=base.fake_backend_name,
        shots=[base.shots],
        mitigations=[base.mitigation],
        zne_scales=base.zne_scales,
        zne_method=base.zne_method,
        seed=base.seed,
    )


def describe_config(cfg: RunConfig) -> str:
    text = f"{cfg.ansatz}(reps={cfg.ansatz_reps})/{cfg.optimizer}, {cfg.mapping}, {cfg.backend}"
    if cfg.mitigation != "none":
        text += f", {cfg.mitigation}"
    return text


def curve_from_results(base: RunConfig, results: list[RunResult]) -> DissociationCurve:
    """Assemble a curve (sorted by bond length) from results of ``sweep_for_config``."""
    ordered = sorted(results, key=lambda r: r.config.bond_length)
    ok = [r for r in ordered if r.status == "ok"]
    if not ok:
        raise ValueError("No successful runs: cannot build a dissociation curve.")
    return DissociationCurve(
        molecule=base.molecule.name,
        label=describe_config(base),
        config_hash=base.config_hash(),
        bond_lengths=[r.config.bond_length for r in ok],
        e_hf=[float(r.e_hf) for r in ok],  # type: ignore[arg-type]
        e_exact=[float(r.e_exact) for r in ok],  # type: ignore[arg-type]
        e_vqe=[r.e_vqe for r in ok],
        backend=base.backend,
        failed=[r.config.bond_length for r in ordered if r.status != "ok"],
    )


def compute_curve(
    base: RunConfig,
    bond_lengths: list[float] | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    force: bool = False,
    max_workers: int = 1,
    show_progress: bool = False,
) -> DissociationCurve:
    """Run (or load from the cache) ``base`` at every bond length and build the curve."""
    lengths = bond_lengths or default_bond_lengths(base.molecule.name)
    summary = run_sweep(
        sweep_for_config(base, lengths),
        results_dir,
        force=force,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    return curve_from_results(base, summary.results)


def save_curve(curve: DissociationCurve, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    path = Path(results_dir) / "curves" / f"{curve.molecule.lower()}_{curve.config_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(curve.to_json())
    return path


def load_curve(path: Path) -> DissociationCurve:
    return DissociationCurve(**json.loads(Path(path).read_text()))


def plot_curve_png(curve: DissociationCurve, path: Path) -> Path:
    """Static figure for the README and slides: energies on top, error and accuracy band below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(curve.bond_lengths)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4]}
    )
    ax1.plot(x, curve.e_hf, "s--", color="#d95f02", label="Hartree-Fock", ms=5)
    ax1.plot(x, curve.e_exact, "-", color="black", lw=2, label="Exact")
    vqe_x = [b for b, v in zip(curve.bond_lengths, curve.e_vqe, strict=True) if v is not None]
    vqe_y = [v for v in curve.e_vqe if v is not None]
    ax1.plot(vqe_x, vqe_y, "o", color="#1b9e77", label=f"VQE ({curve.backend})", ms=7)
    ax1.set_ylabel("Total energy (Hartree)")
    ax1.set_title(f"{curve.molecule} dissociation curve: {curve.label}", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)

    err_x = [b for b, e in zip(curve.bond_lengths, curve.error_mha, strict=True) if e is not None]
    raw_err = [e for e in curve.error_mha if e is not None]
    # Errors below the floor (essentially exact) are drawn at the floor so they stay visible.
    err_y = [max(e, ERROR_FLOOR_MHA) for e in raw_err]
    clipped = any(e < ERROR_FLOOR_MHA for e in raw_err)
    ax2.axhspan(0, CHEMICAL_ACCURACY_HA * 1000, color="#1b9e77", alpha=0.2)
    ax2.text(
        x.max(), CHEMICAL_ACCURACY_HA * 1000 * 1.25, "chemical accuracy (1.6 mHa)",
        fontsize=8, ha="right",
    )  # fmt: skip
    ax2.plot(curve.bond_lengths, curve.hf_error_mha, "s--", color="#d95f02", label="HF error", ms=4)
    vqe_label = "VQE error" + (f" (floored at {ERROR_FLOOR_MHA:g})" if clipped else "")
    ax2.plot(err_x, err_y, "o-", color="#1b9e77", label=vqe_label, ms=5)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=ERROR_FLOOR_MHA / 3)
    ax2.set_xlabel("Bond length (Angstrom)")
    ax2.set_ylabel("Error vs exact (mHa)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_curve_plotly(curve: DissociationCurve):
    """Interactive version of the same figure for the dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08
    )
    fig.add_scatter(
        x=curve.bond_lengths,
        y=curve.e_hf,
        mode="lines+markers",
        name="Hartree-Fock",
        line={"dash": "dash", "color": "#d95f02"},
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=curve.bond_lengths,
        y=curve.e_exact,
        mode="lines",
        name="Exact",
        line={"color": "black", "width": 3},
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=curve.bond_lengths,
        y=curve.e_vqe,
        mode="markers",
        name=f"VQE ({curve.backend})",
        marker={"size": 9, "color": "#1b9e77"},
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=1e-4,
        y1=CHEMICAL_ACCURACY_HA * 1000,
        fillcolor="#1b9e77",
        opacity=0.2,
        line_width=0,
        row=2,
        col=1,
    )
    fig.add_scatter(
        x=curve.bond_lengths,
        y=curve.hf_error_mha,
        mode="lines+markers",
        name="HF error",
        line={"dash": "dash", "color": "#d95f02"},
        showlegend=False,
        row=2,
        col=1,
    )
    fig.add_scatter(
        x=curve.bond_lengths,
        y=curve.error_mha,
        mode="lines+markers",
        name="VQE error",
        line={"color": "#1b9e77"},
        showlegend=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Total energy (Ha)", row=1, col=1)
    fig.update_yaxes(title_text="Error (mHa, log)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Bond length (Angstrom)", row=2, col=1)
    fig.update_layout(title=f"{curve.molecule}: {curve.label}", height=620, template="plotly_white")
    return go.Figure(fig)
