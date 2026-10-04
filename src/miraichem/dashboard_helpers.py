"""Data loading, filtering and figure building for the Streamlit dashboard.

Everything here is a plain function (no Streamlit calls), so it can be unit-tested. The dashboard
only READS saved results, never recomputes anything, which keeps it fast and safe for a live demo.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import plotly.graph_objects as go

from miraichem.analysis.dissociation import DissociationCurve
from miraichem.backends.hardware_eval import HardwareResult
from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.benchmark.ranking import ERROR_FLOOR_MHA, RankedRun, describe
from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR, NON_RUN_DIRS, load_all_results
from miraichem.vqe.result import RunResult

TEAL, ORANGE, GREY, RED = "#1b9e77", "#d95f02", "#7f7f7f", "#c0392b"
MITIGATIONS = ["none", "readout", "zne", "readout+zne"]
CA_MHA = CHEMICAL_ACCURACY_HA * 1000


@dataclass
class DashboardData:
    """Everything the dashboard shows, loaded once from disk."""

    results: list[RunResult] = field(default_factory=list)
    curves: list[DissociationCurve] = field(default_factory=list)
    hardware: list[HardwareResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok_results(self) -> list[RunResult]:
        return [r for r in self.results if r.status == "ok"]

    @property
    def molecules(self) -> list[str]:
        return sorted({r.config.molecule.name for r in self.ok_results})

    def backends_for(self, molecule: str) -> list[str]:
        found = {r.config.backend for r in self.ok_results if r.config.molecule.name == molecule}
        return [b for b in ("ideal", "noisy") if b in found]


def has_run_results(results_dir: Path) -> bool:
    """True if the folder holds at least one saved run (``<molecule>/<hash>.json``)."""
    root = Path(results_dir)
    if not root.is_dir():
        return False
    return any(
        next(sub.glob("*.json"), None) is not None
        for sub in root.glob("*")
        if sub.is_dir() and sub.name not in NON_RUN_DIRS
    )


def load_dashboard_data(
    results_dir: Path = DEFAULT_RESULTS_DIR, hardware_dir: Path | None = None
) -> DashboardData:
    """Load runs, saved curves and hardware comparisons. Unreadable files become warnings."""
    results_dir = Path(results_dir)
    hardware_dir = Path(hardware_dir) if hardware_dir else results_dir / "hardware"
    data = DashboardData()
    try:
        data.results = load_all_results(results_dir)
    except Exception as exc:  # noqa: BLE001 - a bad file must not take the dashboard down
        data.warnings.append(f"Could not read some run results: {exc}")
    for path in sorted((results_dir / "curves").glob("*.json")):
        try:
            data.curves.append(DissociationCurve(**json.loads(path.read_text())))
        except Exception as exc:  # noqa: BLE001
            data.warnings.append(f"Skipped curve file {path.name}: {type(exc).__name__}")
    for path in sorted(hardware_dir.glob("*/*_hw.json")):
        try:
            data.hardware.append(HardwareResult.model_validate_json(path.read_text()))
        except Exception as exc:  # noqa: BLE001
            data.warnings.append(f"Skipped hardware file {path.name}: {type(exc).__name__}")
    return data


def run_label(r: RunResult) -> str:
    return describe(r)


def filter_runs(
    runs: list[RunResult],
    ansatz: list[str] | None = None,
    mapping: list[str] | None = None,
    optimizer: list[str] | None = None,
    mitigation: list[str] | None = None,
) -> list[RunResult]:
    """Keep runs whose settings are in the given lists (None or empty = no restriction)."""
    out = []
    for r in runs:
        c = r.config
        if ansatz and c.ansatz not in ansatz:
            continue
        if mapping and c.mapping not in mapping:
            continue
        if optimizer and c.optimizer not in optimizer:
            continue
        if mitigation and c.mitigation not in mitigation:
            continue
        out.append(r)
    return out


def bond_lengths_for(runs: list[RunResult], molecule: str, backend: str) -> list[float]:
    """Bond lengths available for a molecule/backend, most-populated first."""
    counts: dict[float, int] = defaultdict(int)
    for r in runs:
        if r.config.molecule.name == molecule and r.config.backend == backend:
            counts[r.config.bond_length] += 1
    return sorted(counts, key=lambda b: (-counts[b], b))


# --- figures ---


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def pareto_figure(ranked: list[RankedRun], axis: str = "shots") -> go.Figure:
    """Error vs cost scatter; Pareto-optimal runs are highlighted. ``axis``: shots | gates."""
    on_front = (lambda x: x.on_front_shots) if axis == "shots" else (lambda x: x.on_front_gates)
    cost = (lambda x: x.shots) if axis == "shots" else (lambda x: x.gates)
    xlabel = "Total shots" if axis == "shots" else "Two-qubit gates (logical depth for ideal runs)"
    fig = go.Figure()
    for is_front, name, color, size in (
        (False, "Other runs", GREY, 7),
        (True, "Pareto front", TEAL, 11),
    ):
        pts = [x for x in ranked if on_front(x) == is_front]
        fig.add_scatter(
            x=[cost(x) for x in pts],
            y=[x.error_mha for x in pts],
            mode="markers",
            name=name,
            marker={"size": size, "color": color, "opacity": 0.85},
            text=[f"{x.label}<br>rank {x.rank}" for x in pts],
            hovertemplate="%{text}<br>error %{y:.3g} mHa<br>cost %{x:,.0f}<extra></extra>",
        )
    fig.add_hrect(y0=ERROR_FLOOR_MHA / 3, y1=CA_MHA, fillcolor=TEAL, opacity=0.12, line_width=0)
    fig.update_yaxes(type="log", title="Error vs exact (mHa)")
    fig.update_xaxes(title=xlabel)
    return _layout(fig, f"Error vs {axis}: Pareto front (shaded: chemical accuracy)")


def convergence_figure(r: RunResult, as_error: bool = False) -> go.Figure:
    """Energy (or error) of every optimiser evaluation, against the exact and HF energies."""
    trace = r.convergence_trace
    xs = list(range(1, len(trace) + 1))
    fig = go.Figure()
    exact = r.e_exact if r.e_exact is not None else 0.0

    def conv(values):
        return [max(abs(v - exact) * 1000, ERROR_FLOOR_MHA) for v in values] if as_error else values

    fig.add_scatter(x=xs, y=conv(trace), mode="lines", name="VQE energy", line={"color": TEAL})
    if r.convergence_trace_unmitigated:
        raw = r.convergence_trace_unmitigated
        fig.add_scatter(
            x=list(range(1, len(raw) + 1)),
            y=conv(raw),
            mode="lines",
            name="Unmitigated",
            line={"color": ORANGE, "dash": "dot"},
        )
    if as_error:
        fig.add_hrect(y0=ERROR_FLOOR_MHA / 3, y1=CA_MHA, fillcolor=TEAL, opacity=0.12, line_width=0)
        fig.update_yaxes(type="log", title="Error vs exact (mHa)")
    else:
        if r.e_exact is not None:
            fig.add_hline(y=r.e_exact, line_color="black", annotation_text="exact")
        if r.e_hf is not None:
            fig.add_hline(
                y=r.e_hf, line_color=RED, line_dash="dash", annotation_text="Hartree-Fock"
            )
        fig.update_yaxes(title="Total energy (Ha)")
    fig.update_xaxes(title="Energy evaluation")
    return _layout(fig, f"Convergence: {run_label(r)}")


def mitigation_table(runs: list[RunResult]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Errors (mHa) grouped by (mapping, ansatz) and mitigation mode."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.abs_error_mha is not None:
            groups[(r.config.mapping, r.config.ansatz)][r.config.mitigation].append(r.abs_error_mha)
    return groups


def mitigation_figure(runs: list[RunResult]) -> go.Figure | None:
    """Grouped bars: best error per mitigation mode for each mapping/ansatz (None if no data)."""
    groups = mitigation_table(runs)
    if not any(len(m) > 1 for m in groups.values()):
        return None
    fig = go.Figure()
    colors = {"none": GREY, "readout": "#7570b3", "zne": ORANGE, "readout+zne": TEAL}
    labels = [f"{m} / {a}" for (m, a) in sorted(groups)]
    for mit in MITIGATIONS:
        ys = [min(groups[k][mit]) if groups[k].get(mit) else None for k in sorted(groups)]
        fig.add_bar(name=mit, x=labels, y=ys, marker_color=colors[mit])
    fig.add_hline(y=CA_MHA, line_dash="dash", line_color=TEAL, annotation_text="chemical accuracy")
    fig.update_yaxes(type="log", title="Best error (mHa)")
    fig.update_layout(barmode="group")
    return _layout(fig, "Error mitigation: best error per configuration (lower is better)")


def before_after_figure(runs: list[RunResult]) -> go.Figure | None:
    """Per run: error before (unmitigated) vs after mitigation. Points below the line improved."""
    pts = [
        (r, abs(r.e_vqe_unmitigated - r.e_exact) * 1000, r.abs_error_mha)
        for r in runs
        if r.e_vqe_unmitigated is not None and r.e_exact is not None and r.abs_error_mha is not None
    ]
    if not pts:
        return None
    fig = go.Figure()
    top = max(max(b, a) for _, b, a in pts) * 1.2
    fig.add_scatter(
        x=[ERROR_FLOOR_MHA, top],
        y=[ERROR_FLOOR_MHA, top],
        mode="lines",
        line={"color": GREY, "dash": "dash"},
        name="no change",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=[max(b, ERROR_FLOOR_MHA) for _, b, _ in pts],
        y=[max(a, ERROR_FLOOR_MHA) for _, _, a in pts],
        mode="markers",
        name="mitigated runs",
        marker={"size": 8, "color": TEAL, "opacity": 0.8},
        text=[run_label(r) for r, _, _ in pts],
        hovertemplate="%{text}<br>before %{x:.3g} mHa<br>after %{y:.3g} mHa<extra></extra>",
    )
    fig.update_xaxes(type="log", title="Error before mitigation (mHa)")
    fig.update_yaxes(type="log", title="Error after mitigation (mHa)")
    return _layout(fig, "Before vs after mitigation (below the dashed line = improved)")


def mitigation_summary(runs: list[RunResult]) -> dict[str, float]:
    """Median error (mHa) per mitigation mode across the given runs."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.abs_error_mha is not None:
            by[r.config.mitigation].append(r.abs_error_mha)
    return {m: statistics.median(by[m]) for m in MITIGATIONS if by.get(m)}


def hardware_figure(h: HardwareResult) -> go.Figure:
    """Bars of error vs exact: simulator references next to real-hardware evaluations."""
    labels = ["Ideal circuit", f"Simulator ({h.simulator_mitigation})"]
    errors = [abs(h.e_ideal_at_params - h.e_exact) * 1000, abs(h.e_simulator - h.e_exact) * 1000]
    colors, spreads = [GREY, "#7570b3"], [0.0, 0.0]
    for e in h.evaluations:
        labels.append(f"Hardware ({e.mitigation})")
        errors.append(e.abs_error_mha)
        colors.append(ORANGE)
        spreads.append(e.e_hw_std * 1000)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[max(e, ERROR_FLOOR_MHA) for e in errors],
            marker_color=colors,
            error_y={"type": "data", "array": spreads, "visible": True},
            hovertemplate="%{x}<br>error %{y:.3g} mHa<extra></extra>",
        )
    )
    fig.add_hline(y=CA_MHA, line_dash="dash", line_color=TEAL, annotation_text="chemical accuracy")
    fig.update_yaxes(type="log", title="Error vs exact (mHa); bars: statistical std error")
    return _layout(fig, f"{h.molecule} on {h.backend} (real device vs simulator)", 440)


def hardware_table(h: HardwareResult) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = [
        {"Source": "Exact", "Energy (Ha)": h.e_exact, "Error (mHa)": 0.0, "Kind": "reference"},
        {"Source": "Ideal circuit", "Energy (Ha)": h.e_ideal_at_params,
         "Error (mHa)": abs(h.e_ideal_at_params - h.e_exact) * 1000,
         "Kind": "noiseless simulation"},
        {"Source": f"Simulator ({h.simulator_mitigation})", "Energy (Ha)": h.e_simulator,
         "Error (mHa)": abs(h.e_simulator - h.e_exact) * 1000, "Kind": "noisy simulator"},
    ]  # fmt: skip
    rows += [
        {"Source": f"Hardware ({e.mitigation})", "Energy (Ha)": e.e_hw,
         "Error (mHa)": e.abs_error_mha, "Kind": f"REAL DEVICE, job {e.job_id}"}
        for e in h.evaluations
    ]  # fmt: skip
    return rows


PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#e8f4f0", color="#1b9e77",
        fontname="Helvetica"];
  edge [color="#7f7f7f"];
  mol [label="Molecule\\n(geometry, basis,\\nactive space)"];
  ref [label="Classical reference\\n(HF, FCI/CASCI)"];
  ham [label="Qubit Hamiltonian\\n(Jordan-Wigner /\\nParity)"];
  sweep [label="Sweep of VQE\\nconfigurations\\n(ansatz, optimizer,\\nmitigation, shots)"];
  back [label="Backends\\nideal | noisy fake IBM\\n| real IBM hardware",
        fillcolor="#fdebd9", color="#d95f02"];
  rank [label="Ranking\\nPareto fronts\\nrecommendation"];
  mol -> ref; mol -> ham -> sweep -> back; ref -> rank; back -> rank;
}
"""
