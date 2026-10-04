"""Static matplotlib figures for notebooks and documents (the dashboard uses plotly instead).

Static images always render, including in saved notebooks viewed on GitHub.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.benchmark.ranking import ERROR_FLOOR_MHA, RankedRun
from miraichem.vqe.result import RunResult

TEAL, ORANGE, GREY = "#1b9e77", "#d95f02", "#7f7f7f"


def plot_convergence(result: RunResult, as_error: bool = True) -> Figure:
    """Energy (or error vs exact) after every optimiser evaluation, mitigated and raw if present."""
    exact = result.e_exact or 0.0

    def conv(values: list[float]) -> list[float]:
        if not as_error:
            return values
        return [max(abs(v - exact) * 1000, ERROR_FLOOR_MHA) for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    trace = result.convergence_trace
    ax.plot(range(1, len(trace) + 1), conv(trace), color=TEAL, label="VQE energy")
    raw = result.convergence_trace_unmitigated
    if raw:
        ax.plot(range(1, len(raw) + 1), conv(raw), color=ORANGE, ls=":", label="unmitigated")
    if as_error:
        ax.axhspan(ERROR_FLOOR_MHA / 3, CHEMICAL_ACCURACY_HA * 1000, color=TEAL, alpha=0.15)
        ax.set_yscale("log")
        ax.set_ylabel("Error vs exact (mHa)")
    else:
        ax.axhline(exact, color="black", label="exact")
        ax.set_ylabel("Total energy (Hartree)")
    ax.set_xlabel("Energy evaluation")
    ax.set_title(f"{result.config.ansatz} / {result.config.optimizer} on {result.config.backend}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_pareto(ranked: list[RankedRun], axis: str = "shots") -> Figure:
    """Error vs cost scatter with the Pareto-optimal runs highlighted (axis: shots | gates)."""
    front = (lambda x: x.on_front_shots) if axis == "shots" else (lambda x: x.on_front_gates)
    cost = (lambda x: x.shots) if axis == "shots" else (lambda x: x.gates)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for on_front, label, color, size in (
        (False, "other runs", GREY, 30),
        (True, "Pareto front", TEAL, 70),
    ):
        pts = [x for x in ranked if front(x) == on_front]
        ax.scatter(
            [cost(x) for x in pts],
            [x.error_mha for x in pts],
            c=color,
            s=size,
            label=label,
            alpha=0.85,
        )
    ax.axhspan(ERROR_FLOOR_MHA / 3, CHEMICAL_ACCURACY_HA * 1000, color=TEAL, alpha=0.15)
    ax.set_yscale("log")
    ax.set_ylabel("Error vs exact (mHa)")
    ax.set_xlabel(
        "Total shots" if axis == "shots" else "Two-qubit gates (logical depth for ideal runs)"
    )
    ax.set_title(f"Error vs {axis}: Pareto front (shaded: chemical accuracy)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig
