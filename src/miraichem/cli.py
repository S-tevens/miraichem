"""Command-line interface: ``miraichem reference`` and ``miraichem run``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from miraichem.analysis.dissociation import (
    compute_curve,
    describe_config,
    plot_curve_png,
    save_curve,
)
from miraichem.backends.hardware import (
    DEFAULT_HARDWARE_DIR,
    HardwareAbortedError,
    HardwareDisabledError,
    HardwarePlan,
    fetch_job,
    get_service,
)
from miraichem.backends.hardware_eval import (
    load_source_run,
    plan_evaluation,
    run_hardware_evaluation,
    save_hardware_result,
)
from miraichem.benchmark.ranking import Weights, recommend
from miraichem.benchmark.ranking_scan import recommend_across_scan
from miraichem.benchmark.report import load_hardware_results, write_report
from miraichem.benchmark.storage import (
    DEFAULT_RESULTS_DIR,
    export_summary,
    load_all_results,
    save_result,
)
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
def sqd(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML, e.g. configs/sqd/n2.yaml")],
    bond_length: Annotated[float, typer.Option(help="Bond length in Angstrom")] = 1.0977,
    n_reps: int = 1,
    shots: int = 20000,
    readout_noise: Annotated[
        float, typer.Option(help="Bit-flip probability per sampled bit")
    ] = 0.0,
    iterations: int = 3,
    samples_per_batch: int = 300,
    num_batches: int = 3,
    seed: int = 42,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Run sample-based quantum diagonalization (simulated sampling) and save the result."""
    from miraichem.sqd import SqdConfig, run_sqd
    from miraichem.sqd.storage import save_sqd_result

    cfg = SqdConfig(
        molecule=_load_molecule(molecule),
        bond_length=bond_length,
        n_reps=n_reps,
        shots=shots,
        readout_noise=readout_noise,
        iterations=iterations,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        seed=seed,
    )
    result = run_sqd(cfg)
    path = save_sqd_result(result, results_dir)
    if result.status == "failed":
        console.print(f"[red]Run failed:[/red] {result.error_message}\nSaved to {path}")
        raise typer.Exit(1)
    table = Table(title=f"{cfg.molecule.name} SQD, {result.num_qubits} qubits")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("SQD energy (Ha)", f"{result.e_sqd:.6f}")
    table.add_row("Exact FCI energy (Ha)", f"{result.e_exact:.6f}")
    table.add_row("Hartree-Fock error (mHa)", f"{result.hf_error_mha:.3f}")
    table.add_row("SQD error (mHa)", f"{result.abs_error_mha:.3f}")
    table.add_row(
        "Error per round (mHa)", ", ".join(f"{e:.2f}" for e in result.error_per_iteration_mha)
    )
    table.add_row("Chemical accuracy", "yes" if result.within_chemical_accuracy else "no")
    table.add_row("Distinct samples", str(result.n_distinct_samples))
    table.add_row("Right electron count", str(result.n_postselected))
    table.add_row(
        "Subspace / full CI space", f"{result.subspace_dim:,} / {result.n_determinants:,}"
    )
    table.add_row("Wall time (s)", f"{result.wall_time_s:.1f}")
    console.print(table)
    console.print(f"Saved to {path}")


@app.command("sqd-hardware")
def sqd_hardware(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML, e.g. configs/sqd/n2.yaml")],
    bond_length: Annotated[float, typer.Option(help="Bond length in Angstrom")] = 1.0977,
    backend_name: Annotated[
        str | None, typer.Option(help="Device name; default least busy")
    ] = None,
    n_reps: int = 1,
    shots: int = 10000,
    iterations: int = 3,
    samples_per_batch: int = 300,
    num_batches: int = 3,
    seed: int = 42,
    max_qpu_seconds: Annotated[
        float, typer.Option(help="Refuse plans estimated above this")
    ] = 120.0,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    allow_hardware: Annotated[bool, typer.Option(help="REQUIRED to submit a real job")] = False,
) -> None:
    """Sample the SQD circuit on a real IBM device, then finish the calculation classically.

    Without --allow-hardware this is a dry run: it transpiles the circuit for the device, prints the
    QPU-time estimate, and submits nothing.
    """
    from miraichem.backends.hardware import HardwareAbortedError, HardwareDisabledError
    from miraichem.sqd import SqdConfig
    from miraichem.sqd.hardware import CONFIRM_PHRASE, collect, prepare_plan, submit
    from miraichem.sqd.storage import save_sqd_result

    cfg = SqdConfig(
        molecule=_load_molecule(molecule),
        bond_length=bond_length,
        n_reps=n_reps,
        shots=shots,
        iterations=iterations,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        seed=seed,
    )
    console.print("Preparing the circuit (no QPU time is used)...")
    plan = prepare_plan(cfg, backend_name=backend_name, max_qpu_seconds=max_qpu_seconds)
    p = plan.as_dict()
    console.print(f"Backend          : {p['backend']}")
    console.print(f"Qubits           : {p['qubits']} ({p['qubits_used']} used after routing)")
    console.print(f"Two-qubit gates  : {p['two_qubit_gates']}  (depth {p['depth']})")
    console.print(f"Shots            : {p['shots']}")
    console.print(f"Estimated QPU time: ~{p['estimated_qpu_s']:.0f} s (upper-bound estimate)")
    console.print(f"Classical post-processing: ~{p['estimated_post_s']:.0f} s")
    if p["remaining_qpu_s"] is not None:
        console.print(f"QPU time remaining: {p['remaining_qpu_s']:.0f} s of {p['budget_s']:.0f} s")
    if p["pending_jobs"] is not None:
        console.print(f"Jobs queued on the device: {p['pending_jobs']} (wait time not included)")
    for problem in plan.problems:
        console.print(f"[red]{problem}[/red]")
    if not allow_hardware:
        console.print(
            "\nDry run: nothing was submitted. Add [bold]--allow-hardware[/bold] to submit."
        )
        return
    if plan.problems:
        raise typer.Exit(1)
    if not typer.confirm("Submit this job to REAL hardware and use QPU time?", default=False):
        console.print("Not confirmed; nothing was sent.")
        raise typer.Exit(1)
    try:
        job = submit(plan, CONFIRM_PHRASE, allow_hardware=True, hardware_dir=hardware_dir)
    except (HardwareAbortedError, HardwareDisabledError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"Submitted job {job.job_id()}. If this session ends, finish it later with "
        f"`miraichem sqd-hardware-fetch {job.job_id()}`."
    )
    result = collect(job, plan, on_status=lambda s: console.print(f"  IBM job: {s.lower()}"))
    path = save_sqd_result(result, results_dir)
    _print_sqd_result(result, f"{cfg.molecule.name} SQD on {p['backend']}")
    console.print(f"Saved to {path}")


@app.command("sqd-hardware-fetch")
def sqd_hardware_fetch(
    job_id: str,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
) -> None:
    """Finish an earlier hardware SQD job from its saved record (uses no QPU time)."""
    from miraichem.sqd.hardware import finalize_saved_job
    from miraichem.sqd.storage import save_sqd_result

    result = finalize_saved_job(
        job_id,
        hardware_dir=hardware_dir,
        on_status=lambda s: console.print(f"  IBM job: {s.lower()}"),
    )
    path = save_sqd_result(result, results_dir)
    _print_sqd_result(result, f"{result.config.molecule.name} SQD on {result.hardware['backend']}")
    console.print(f"Saved to {path}")


def _print_sqd_result(result, title: str) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("SQD energy (Ha)", f"{result.e_sqd:.6f}")
    table.add_row("Hartree-Fock energy (Ha)", f"{result.e_hf:.6f}")
    table.add_row("CCSD energy (Ha)", f"{result.e_ccsd:.6f}")
    if result.e_exact is not None:
        table.add_row("Exact FCI energy (Ha)", f"{result.e_exact:.6f}")
        table.add_row("SQD error (mHa)", f"{result.abs_error_mha:.3f}")
    table.add_row("Below Hartree-Fock (mHa)", f"{result.below_hf_mha:.1f}")
    table.add_row("Below CCSD (mHa)", f"{result.below_ccsd_mha:.1f}")
    table.add_row("Subspace / full space", f"{result.subspace_dim:,} / {result.n_determinants:,}")
    hw = result.hardware or {}
    if hw.get("qpu_seconds") is not None:
        table.add_row("QPU time used (s)", f"{hw['qpu_seconds']:.1f}")
    console.print(table)


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


@app.command()
def rank(
    molecule: Annotated[str, typer.Option(help="Molecule name, e.g. h2")],
    backend: Annotated[str, typer.Option(help="ideal | noisy")] = "noisy",
    bond_length: Annotated[
        float | None, typer.Option(help="Angstrom; default: most common")
    ] = None,
    top: Annotated[int, typer.Option(help="Rows to show")] = 10,
    weight_accuracy: float = 0.7,
    weight_gates: float = 0.2,
    weight_shots: float = 0.1,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    scan: Annotated[
        bool,
        typer.Option(help="Rank across bond-length scans (worst case) instead of one geometry"),
    ] = False,
    replicated: Annotated[
        bool,
        typer.Option(help="Rank by mean error over seeds and recommend what you can trust"),
    ] = False,
    runs_csv: Annotated[
        Path | None, typer.Option(help="With --replicated: read per-seed rows from this CSV")
    ] = None,
    min_seeds: Annotated[int, typer.Option(help="With --replicated: minimum seeds")] = 3,
) -> None:
    """Rank saved runs for a molecule and backend and print the recommended configuration."""
    if replicated:
        _rank_replicated(molecule, backend, bond_length, top, results_dir, runs_csv, min_seeds)
        return
    results = load_all_results(results_dir, molecule)
    weights = Weights(weight_accuracy, weight_gates, weight_shots)
    if scan:
        _rank_scan(results, molecule, backend, weights, top)
        return
    try:
        rec = recommend(results, molecule, backend, weights, bond_length)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title=f"{molecule} on {backend}: leaderboard (top {top} of {rec.n_candidates})")
    for col in ("#", "Configuration", "Error mHa", "CA", "2q gates", "Shots", "Score", "Pareto"):
        table.add_column(col, justify="left" if col == "Configuration" else "right")
    for x in rec.ranked[:top]:
        marks = ("S" if x.on_front_shots else "") + ("G" if x.on_front_gates else "")
        gates = str(x.result.two_qubit_gates) if x.result.two_qubit_gates is not None else "-"
        table.add_row(
            str(x.rank),
            x.label,
            f"{x.error_mha:.3g}",
            "yes" if x.within_chemical_accuracy else "no",
            gates,
            f"{x.shots:,}",
            f"{x.score:.3f}",
            marks,
        )
    console.print(table)
    console.print("Pareto: S = error vs shots front, G = error vs two-qubit gates front.\n")
    console.print(rec.justification)


def _rank_replicated(
    molecule: str,
    backend: str,
    bond_length: float | None,
    top: int,
    results_dir: Path,
    runs_csv: Path | None,
    min_seeds: int,
) -> None:
    from miraichem.benchmark.ranking_replicated import recommend_replicated
    from miraichem.benchmark.replication import read_runs_csv, runs_frame

    runs = read_runs_csv(runs_csv) if runs_csv else runs_frame(load_all_results(results_dir))
    try:
        rec = recommend_replicated(runs, molecule, backend, bond_length, min_seeds)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title=f"{molecule} on {backend}: replicated leaderboard (top {top})")
    for col in ("#", "Configuration", "Seeds", "Mean ± 95% CI", "Worst", "<1.6 mHa", "2q gates"):
        table.add_column(col, justify="left" if col == "Configuration" else "right", no_wrap=True)
    for c in rec.ranked[:top]:
        mark = " *" if c is rec.best else ""
        table.add_row(
            str(c.rank),
            c.label + mark,
            str(c.n_seeds),
            f"{c.mean:.1f} ± {c.ci95:.1f}",
            f"{c.worst:.1f}",
            f"{c.n_chem}/{c.n_seeds}",
            f"{c.gates:.0f}",
        )
    console.print(table)
    console.print("* = recommended.\n")
    console.print(rec.justification)


@app.command()
def devices(
    molecule: Annotated[str, typer.Option(help="Molecule name, e.g. H2")] = "H2",
    backend: str = "noisy",
    min_seeds: Annotated[int, typer.Option(help="Minimum seeds per configuration")] = 5,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_csv: Annotated[Path | None, typer.Option(help="Read per-seed rows from this CSV")] = None,
) -> None:
    """Does the recommendation transfer across fake IBM devices? (needs a multi-device sweep)"""
    from miraichem.benchmark.devices import compare_devices, summarize_agreement
    from miraichem.benchmark.replication import read_runs_csv, runs_frame

    runs = read_runs_csv(runs_csv) if runs_csv else runs_frame(load_all_results(results_dir))
    try:
        cmp = compare_devices(runs, molecule, backend, min_seeds=min_seeds)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    t = Table(title=f"{molecule}: mean error (mHa) over seeds, per device")
    t.add_column("Configuration", no_wrap=True)
    for d in cmp.devices:
        t.add_column(d.replace("Fake", ""), justify="right", no_wrap=True)
    for name, row in cmp.means.iterrows():
        t.add_row(str(name), *[f"{row[d]:.1f}" if row[d] == row[d] else "-" for d in cmp.devices])
    console.print(t)
    console.print("\nRank correlation (Spearman) between devices:")
    console.print(cmp.spearman.round(2).to_string())
    console.print("\nRecommended per device:")
    for d, rec in cmp.recommendations.items():
        console.print(
            f"  {d}: {rec.best.label}  ({rec.best.mean:.1f} ± {rec.best.std:.1f} mHa, "
            f"{rec.best.n_chem}/{rec.best.n_seeds} within 1.6 mHa)"
        )
    console.print("\n" + summarize_agreement(cmp))


@app.command()
def replicate(
    molecule: Annotated[str, typer.Option(help="Molecule name, e.g. H2")] = "H2",
    backend: Annotated[
        str, typer.Option(help="Only this backend (ideal, noisy), or 'all'")
    ] = "noisy",
    min_seeds: Annotated[int, typer.Option(help="Ignore configs with fewer seeds")] = 3,
    top: Annotated[int, typer.Option(help="Rows to show")] = 15,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runs_csv: Annotated[
        Path | None, typer.Option(help="Read per-run rows from this CSV instead of result files")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Also write the summary table to this CSV")
    ] = None,
    export_runs: Annotated[
        Path | None, typer.Option(help="Write the per-run rows (one per seed) to this CSV")
    ] = None,
) -> None:
    """Mean +/- std error over seeds for every replicated configuration."""
    from miraichem.benchmark.replication import (
        GROUP_COLS,
        config_label,
        normalize_runs,
        read_runs_csv,
        runs_frame,
        summarize,
    )

    df = read_runs_csv(runs_csv) if runs_csv else runs_frame(load_all_results(results_dir))
    table = summarize(df, molecule, min_seeds=min_seeds)
    if table.empty:
        console.print("[yellow]No replicated configurations found.[/yellow]")
        raise typer.Exit(1)
    if backend != "all":
        table = table[table.backend == backend]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        # every molecule and backend, so the published table matches the published per-seed rows
        summarize(df, None, min_seeds=min_seeds).to_csv(out, index=False)
    if export_runs:
        # every replicated configuration of every molecule, so one file holds the whole study
        keys = summarize(df, None, min_seeds=min_seeds)[GROUP_COLS]
        keep = normalize_runs(df).merge(keys, on=GROUP_COLS)
        export_runs.parent.mkdir(parents=True, exist_ok=True)
        keep.sort_values([*GROUP_COLS, "seed"]).to_csv(export_runs, index=False)
    t = Table(title=f"{molecule} replicated over seeds ({backend}), best mean error first")
    t.add_column("Configuration", no_wrap=True)
    for col in ("Seeds", "Mean", "Std", "Median", "Worst", "<1.6 mHa"):
        t.add_column(col, justify="right", no_wrap=True)
    for _, r in table.head(top).iterrows():
        t.add_row(
            config_label(r, detail=True),
            str(r["n_seeds"]),
            f"{r['mean_mHa']:.2f}",
            f"{r['std_mHa']:.2f}",
            f"{r['median_mHa']:.2f}",
            f"{r['max_mHa']:.2f}",
            f"{r['frac_chem_acc']:.0%}",
        )
    console.print(t)
    console.print("Errors in mHa vs the exact energy; chemical accuracy is 1.6 mHa.")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Address to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    results_dir: Annotated[
        Path, typer.Option(help="Where your runs are saved")
    ] = DEFAULT_RESULTS_DIR,
    published_dir: Annotated[Path, typer.Option(help="Published results")] = Path(
        "results/published"
    ),
    configs_dir: Annotated[Path, typer.Option(help="Molecule and sweep configs")] = Path("configs"),
    static_dir: Annotated[Path, typer.Option(help="Built frontend")] = Path("frontend/dist"),
    allow_remote: Annotated[
        bool, typer.Option(help="Allow listening on a non-local address (the app can start jobs)")
    ] = False,
    allow_hardware: Annotated[
        bool,
        typer.Option(
            help="Let the app submit SQD sampling jobs to real IBM hardware. Each one still "
            "needs a plan and a typed confirmation."
        ),
    ] = False,
    max_qpu_seconds: Annotated[
        float, typer.Option(help="Refuse hardware plans estimated above this much QPU time")
    ] = 120.0,
) -> None:
    """Start the MiraiChem app: the web interface plus the API behind it."""
    import uvicorn

    from miraichem.api import create_app

    if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
        console.print(
            "[red]Refusing to listen on a non-local address: the app can start simulator jobs "
            "and has no login. Use --allow-remote only on a network you trust.[/red]"
        )
        raise typer.Exit(1)
    if not (static_dir / "index.html").is_file():
        console.print(
            f"[yellow]No built frontend at {static_dir}. The API still works; build the UI with "
            "`npm --prefix frontend run build`, or run it with "
            "`npm --prefix frontend run dev`.[/yellow]"
        )
    api = create_app(
        results_dir,
        published_dir,
        configs_dir,
        static_dir,
        allow_hardware=allow_hardware,
        max_qpu_seconds=max_qpu_seconds,
    )
    if allow_hardware:
        console.print(
            "[yellow]Real hardware is ENABLED for SQD "
            f"(limit {max_qpu_seconds:.0f} s of QPU time per run). "
            "The app can submit jobs after a plan and a typed confirmation.[/yellow]"
        )
    console.print(f"MiraiChem app on http://{host}:{port}  (API docs at /api/docs)")
    uvicorn.run(api, host=host, port=port, log_level="warning")


predictor_app = typer.Typer(help="Predict a noisy run's error before running it.")
app.add_typer(predictor_app, name="predictor")

DEFAULT_PREDICTOR = Path("results/models/error_predictor.json")


@predictor_app.command("train")
def predictor_train(
    results_dir: Annotated[
        Path, typer.Option(help="Saved runs to learn from")
    ] = DEFAULT_RESULTS_DIR,
    out: Annotated[Path, typer.Option(help="Where to save the trained model")] = DEFAULT_PREDICTOR,
    trees: Annotated[int, typer.Option(help="Trees in the forest")] = 60,
) -> None:
    """Train the error predictor from saved noisy runs; reports its accuracy on unseen ones."""
    from miraichem.predict import ErrorPredictor

    results = load_all_results(results_dir)
    try:
        model = ErrorPredictor.train(
            results, progress=lambda m: console.print(f"[dim]{m}...[/dim]"), n_trees=trees
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    path = model.save(out)
    console.print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB).")
    _print_predictor_report(model.report)


def _print_predictor_report(rep: dict) -> None:
    cv, b1, b2 = rep["cross_validation"], rep["baseline_mean"], rep["baseline_per_molecule"]
    t = Table(
        title=(
            f"Validated on unseen configurations "
            f"({rep['n_runs']} runs, {rep['n_configurations']} configs)"
        )
    )
    for col in ("Model", "Typical error factor", "MAE (log10)", "R2"):
        t.add_column(col, justify="left" if col == "Model" else "right")
    for name, m in (("Predictor", cv), ("Per-molecule mean", b2), ("Global mean", b1)):
        t.add_row(name, f"{m['typical_factor']:.2f}x", f"{m['mae_log10']:.3f}", f"{m['r2']:.2f}")
    console.print(t)
    iv = rep["interval"]
    console.print(
        f"80% interval: really contained the answer {iv['coverage']:.0%} of the time "
        f"(calibrated on other folds)."
    )
    for h in rep["device_holdout"]:
        note = "" if h["fair"] else " (not a fair test: most of the data came from this device)"
        console.print(
            f"  held-out device {h['device']}: typical factor {h['typical_factor']:.2f}x "
            f"(n={h['n']}){note}"
        )
    top = list(rep["importance"].items())[:6]
    console.print("Most relied on: " + ", ".join(f"{k} (+{v:.3f})" for k, v in top))


@predictor_app.command("show")
def predictor_show(
    model: Annotated[Path, typer.Option(help="Trained model file")] = DEFAULT_PREDICTOR,
) -> None:
    """Print how well a trained predictor did when it was validated."""
    from miraichem.predict import ErrorPredictor

    if not model.is_file():
        console.print(
            f"[red]No model at {model}. Train one with `miraichem predictor train`.[/red]"
        )
        raise typer.Exit(1)
    _print_predictor_report(ErrorPredictor.load(model).report)


@app.command("export-demo")
def export_demo_cmd(
    out: Annotated[Path, typer.Option(help="Folder the demo build reads its data from")] = Path(
        "frontend/dist-demo/demo"
    ),
    published_dir: Annotated[Path, typer.Option(help="Published results")] = Path(
        "results/published"
    ),
    configs_dir: Annotated[Path, typer.Option(help="Molecule and sweep configs")] = Path("configs"),
) -> None:
    """Write the data for the offline (read-only) demo build of the web app."""
    from miraichem.api.demo import export_demo

    counts = export_demo(out, published_dir, configs_dir)
    size = sum(f.stat().st_size for f in out.rglob("*.json")) / 1e6
    console.print(
        f"Wrote {sum(counts.values())} files ({size:.1f} MB) to {out}: "
        f"{counts['views']} views, {counts['rank']} rankings, {counts['scan']} scans, "
        f"{counts['runs']} runs."
    )


@app.command()
def report(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    output: Annotated[Path, typer.Option(help="Markdown file to write")] = Path("docs/RESULTS.md"),
    export_dir: Annotated[
        Path, typer.Option(help="Where to write summary.csv and summary.parquet")
    ] = DEFAULT_RESULTS_DIR / "summary",
) -> None:
    """Write docs/RESULTS.md and a CSV/Parquet summary from the saved results."""
    results = load_all_results(results_dir)
    if not results:
        console.print(f"[red]No saved results in {results_dir}/. Run a sweep first.[/red]")
        raise typer.Exit(1)
    replication = None
    runs_csv = Path(results_dir) / "summary" / "replication_runs.csv"
    if runs_csv.is_file():
        from miraichem.benchmark.replication import read_runs_csv

        replication = read_runs_csv(runs_csv)
    path = write_report(
        results, output, load_hardware_results(hardware_dir), replication=replication
    )
    csv_path, parquet_path = export_summary(results, export_dir)
    console.print(f"Wrote {path} from {len(results)} runs.\nSummary: {csv_path} and {parquet_path}")


def _rank_scan(results: list, molecule: str, backend: str, weights: Weights, top: int) -> None:
    try:
        rec = recommend_across_scan(results, molecule, backend, weights)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title=f"{molecule} on {backend}: scan leaderboard (top {top} of {rec.n_configs})")
    for col in ("#", "Configuration", "Points", "Worst mHa", "Mean mHa", "Coverage", "Score"):
        table.add_column(col, justify="left" if col == "Configuration" else "right")
    for s in rec.ranked[:top]:
        table.add_row(
            str(s.rank),
            s.label,
            str(len(s.bond_lengths)),
            f"{s.worst_error_mha:.3g}",
            f"{s.mean_error_mha:.3g}",
            f"{s.coverage:.0%}",
            f"{s.score:.3f}",
        )
    console.print(table)
    console.print("Coverage = share of scanned bond lengths within chemical accuracy (1.6 mHa).\n")
    console.print(rec.justification)


@app.command()
def curve(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML")],
    config_hash: Annotated[
        str | None, typer.Option(help="Saved run whose configuration to scan; default: recommended")
    ] = None,
    backend: Annotated[
        str, typer.Option(help="Used with the recommendation (ideal | noisy)")
    ] = "ideal",
    bond_lengths: Annotated[
        str | None, typer.Option(help="Comma-separated Angstrom values")
    ] = None,
    max_workers: int = 1,
    force: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = Path("docs/figures"),
) -> None:
    """Scan a configuration over bond lengths and plot HF vs VQE vs exact (PNG + saved data)."""
    mol = _load_molecule(molecule)
    if config_hash:
        base = load_source_run(config_hash, mol.name, results_dir).config
    else:
        try:
            base = recommend(
                load_all_results(results_dir, mol.name), mol.name, backend
            ).best.result.config
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    lengths = [float(x) for x in bond_lengths.split(",")] if bond_lengths else None
    result_curve = compute_curve(base, lengths, results_dir, force, max_workers, show_progress=True)
    data_path = save_curve(result_curve, results_dir)
    png = plot_curve_png(
        result_curve,
        output_dir / f"dissociation_{mol.name.lower()}_{base.backend}_{base.config_hash()}.png",
    )
    worst = result_curve.max_vqe_error_mha
    console.print(
        f"{mol.name}: {describe_config(base)}\n"
        f"  points: {len(result_curve.bond_lengths)} ({len(result_curve.failed)} failed), "
        f"within chemical accuracy at {result_curve.fraction_chemically_accurate:.0%} of them, "
        f"worst VQE error {worst:.3g} mHa vs worst HF error "
        f"{max(result_curve.hf_error_mha):.3g} mHa\n"
        f"  figure: {png}\n  data: {data_path}"
    )


def _print_plan(plan: HardwarePlan) -> None:
    console.print("[bold yellow]Real hardware cost estimate[/bold yellow]")
    for line in plan.lines():
        console.print("  " + line)


@app.command()
def hardware(
    molecule: Annotated[Path, typer.Option(help="Molecule YAML")],
    config_hash: Annotated[str, typer.Option(help="Hash of a saved simulator run to evaluate")],
    backend_name: Annotated[
        str | None, typer.Option(help="Device name; default least busy")
    ] = None,
    shots: int = 4096,
    modes: Annotated[
        str, typer.Option(help="Comma-separated mitigation modes")
    ] = "none,readout+zne",
    results_dir: Path = DEFAULT_RESULTS_DIR,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    allow_hardware: Annotated[bool, typer.Option(help="REQUIRED to submit real jobs")] = False,
) -> None:
    """Evaluate a saved run's optimal parameters on real IBM hardware (fixed-parameter mode).

    Without --allow-hardware this is a dry run: it prints the cost estimate and sends nothing.
    """
    mol = _load_molecule(molecule)
    source = load_source_run(config_hash, mol.name, results_dir)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    plan = plan_evaluation(source, mode_list, shots, backend_name)  # type: ignore[arg-type]
    _print_plan(plan)
    if not allow_hardware:
        console.print(
            "\nDry run: nothing was submitted. Add [bold]--allow-hardware[/bold] to submit."
        )
        return

    def confirm(p: HardwarePlan) -> bool:
        return typer.confirm("Submit these jobs to REAL hardware and use QPU time?", default=False)

    try:
        result = run_hardware_evaluation(
            source,
            mode_list,  # type: ignore[arg-type]
            shots,
            confirm,
            backend_name,
            hardware_dir=hardware_dir,
        )
    except (HardwareAbortedError, HardwareDisabledError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    path = save_hardware_result(result, hardware_dir)
    table = Table(title=f"{result.molecule} on {result.backend} (error vs exact, mHa)")
    table.add_column("Source")
    table.add_column("Energy (Ha)", justify="right")
    table.add_column("Error (mHa)", justify="right")
    table.add_row("Exact", f"{result.e_exact:.6f}", "0")
    table.add_row("Ideal circuit", f"{result.e_ideal_at_params:.6f}", "")
    table.add_row(
        f"Simulator ({result.simulator_mitigation})",
        f"{result.e_simulator:.6f}",
        f"{abs(result.e_simulator - result.e_exact) * 1000:.1f}",
    )
    for ev in result.evaluations:
        table.add_row(f"Hardware ({ev.mitigation})", f"{ev.e_hw:.6f}", f"{ev.abs_error_mha:.1f}")
    console.print(table)
    console.print(f"Saved to {path}")


@app.command("hardware-fetch")
def hardware_fetch(job_id: str) -> None:
    """Re-fetch a finished hardware job by ID (does not use QPU time)."""
    info = fetch_job(get_service(), job_id)
    console.print(info)


if __name__ == "__main__":
    app()
