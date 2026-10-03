"""Command-line interface: ``miraichem reference`` and ``miraichem run``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR, save_result
from miraichem.benchmark.sweep import load_sweep_config, run_sweep
from miraichem.chemistry.classical import compute_reference
from miraichem.config import MoleculeConfig, RunConfig
from miraichem.vqe.runner import run_vqe

app = typer.Typer(help="MiraiChem: benchmark VQE configurations on ideal and noisy backends.")
console = Console()


def _load_molecule(path: Path) -> MoleculeConfig:
    return MoleculeConfig(**yaml.safe_load(path.read_text()))


@app.command()
def reference(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML, e.g. configs/molecules/h2.yaml")],
    bond_length: Annotated[float, typer.Option(help="Bond length in Angstrom")] = 0.735,
) -> None:
    """Print classical HF and exact (FCI/CASCI) total energies in Hartree."""
    mol = _load_molecule(molecule)
    ref = compute_reference(mol, bond_length)
    table = Table(title=f"{mol.name} at {bond_length} A (total energies, Hartree)")
    table.add_column("Quantity")
    table.add_column("Value", justify="right")
    table.add_row("HF", f"{ref.e_hf:.8f}")
    table.add_row(f"Exact ({ref.exact_method})", f"{ref.e_exact:.8f}")
    table.add_row("Nuclear repulsion", f"{ref.e_nuclear_repulsion:.8f}")
    table.add_row("Qubits (JW)", str(ref.n_qubits_expected))
    console.print(table)


@app.command()
def run(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML")],
    bond_length: Annotated[float, typer.Option(help="Bond length in Angstrom")] = 0.735,
    mapping: str = "jordan_wigner",
    ansatz: str = "uccsd",
    ansatz_reps: int = 1,
    optimizer: str = "cobyla",
    maxiter: int = 200,
    backend: str = "ideal",
    fake_backend_name: str = "FakeSherbrooke",
    shots: int = 4096,
    mitigation: str = "none",
    seed: int = 42,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Run one VQE configuration and save the result JSON."""
    cfg = RunConfig(
        molecule=_load_molecule(molecule),
        bond_length=bond_length,
        mapping=mapping,  # type: ignore[arg-type]
        ansatz=ansatz,  # type: ignore[arg-type]
        ansatz_reps=ansatz_reps,
        optimizer=optimizer,  # type: ignore[arg-type]
        maxiter=maxiter,
        backend=backend,  # type: ignore[arg-type]
        fake_backend_name=fake_backend_name,
        shots=shots,
        mitigation=mitigation,  # type: ignore[arg-type]
        seed=seed,
    )
    result = run_vqe(cfg)
    path = save_result(result, results_dir)
    if result.status == "failed":
        console.print(f"[red]Run failed:[/red] {result.error_message}\nSaved to {path}")
        raise typer.Exit(1)
    table = Table(title=f"{cfg.molecule.name} {cfg.ansatz}/{cfg.optimizer} on {cfg.backend}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("VQE energy (Ha)", f"{result.e_vqe:.6f}")
    table.add_row("Exact energy (Ha)", f"{result.e_exact:.6f}")
    table.add_row("HF energy (Ha)", f"{result.e_hf:.6f}")
    if result.e_vqe_unmitigated is not None:
        table.add_row("Unmitigated energy (Ha)", f"{result.e_vqe_unmitigated:.6f}")
    table.add_row("Error (mHa)", f"{result.abs_error_mha:.3f}")
    table.add_row("Chemical accuracy", "yes" if result.within_chemical_accuracy else "no")
    table.add_row("Evaluations", str(result.n_function_evals))
    table.add_row("Wall time (s)", f"{result.wall_time_s:.1f}")
    console.print(table)
    console.print(f"Saved to {path}")


@app.command()
def sweep(
    config: Annotated[Path, typer.Argument(help="Sweep YAML, e.g. configs/sweeps/quick_h2.yaml")],
    force: Annotated[bool, typer.Option(help="Rerun even if a result is already saved")] = False,
    max_workers: Annotated[int, typer.Option(help="Parallel processes (simulator runs)")] = 1,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Run every configuration in a sweep; saved results are skipped unless --force."""
    sweep_cfg = load_sweep_config(config)
    summary = run_sweep(sweep_cfg, results_dir, force=force, max_workers=max_workers)
    console.print(
        f"[bold]{summary.n_total}[/bold] runs: {summary.n_ran} ran "
        f"({summary.n_failed} failed), {summary.n_cached} cached. Results in {results_dir}/"
    )
    for r in summary.results:
        if r.status == "failed":
            console.print(f"[red]failed[/red] {r.config_hash}: {r.error_message}")
    if summary.n_failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
