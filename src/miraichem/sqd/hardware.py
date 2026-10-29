"""Real-hardware sampling for SQD, behind the same safeguards as the VQE hardware path.

SQD needs only ONE circuit run: sample a LUCJ circuit on a device; everything else is classical.
That makes it cheap in QPU time (seconds), but the free plan is still small, so the rules are the
same as for ``backends/hardware.py``:

  1. Nothing is created unless ``allow_hardware`` is True (``miraichem serve --allow-hardware``, or
     ``--allow-hardware`` on the command line).
  2. A plan is made first. It transpiles the circuit for the chosen device and estimates the QPU
     time, and it uses no QPU time itself. Submitting needs that plan plus a typed confirmation.
  3. A plan whose estimate exceeds the server's per-run cap, or the QPU time that remains, cannot be
     submitted.
  4. The job ID and everything needed to rebuild the result are written to disk the moment the job
     is submitted, so a crash or a closed browser never loses a job.

The circuit differs from the simulated one: its orbital couplings are restricted to what fits an
IBM heavy-hex device (``device_interaction_pairs``). Results therefore also record what the same
circuit gives on an ideal simulator, when that is small enough to simulate.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

import ffsim
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_sqd.counts import counts_to_arrays

from miraichem.backends.hardware import (
    DEFAULT_HARDWARE_DIR,
    FREE_PLAN_BUDGET_S,
    HardwareAbortedError,
    HardwareDisabledError,
    get_service,
    record_job,
    remaining_qpu_seconds,
    resolve_backend,
)
from miraichem.sqd.config import SqdConfig
from miraichem.sqd.pipeline import diagonalise_samples, library_versions
from miraichem.sqd.problem import SqdProblem, build_problem, n_determinants
from miraichem.sqd.result import SqdResult
from miraichem.sqd.sampler import MAX_DETERMINANTS, device_interaction_pairs, sample_lucj
from miraichem.vqe.runner import git_commit

CONFIRM_PHRASE = "SUBMIT TO REAL HARDWARE"
PLAN_TTL_S = 1800.0  # a plan is only valid for half an hour (queue and budget change)
REP_DELAY_S = 250e-6  # IBM's default delay between shots
JOB_OVERHEAD_S = 3.0  # fixed cost per job, as in the VQE hardware plan
SAFETY_FACTOR = 1.5
DEFAULT_MAX_QPU_S = 120.0  # per-run cap; the free plan has 600 s per period in total
FINAL_STATES = {"DONE", "CANCELLED", "ERROR"}


# ------------------------------------------------------------------ circuit and estimate


def build_circuit(problem: SqdProblem, n_reps: int) -> QuantumCircuit:
    """LUCJ circuit on the Hartree-Fock state with device-friendly couplings, then measured."""
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        problem.ccsd_t2,
        t1=problem.ccsd_t1,
        n_reps=n_reps,
        interaction_pairs=device_interaction_pairs(problem.norb),
    )
    qubits = QuantumRegister(2 * problem.norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(problem.norb, problem.nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), qubits)
    circuit.measure_all()
    return circuit


def transpile_for(circuit: QuantumCircuit, device, optimization_level: int = 3, seed: int = 1):
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=device, seed_transpiler=seed
    )
    pm.pre_init = ffsim.qiskit.PRE_INIT
    return pm.run(circuit)


def circuit_stats(isa: QuantumCircuit, device) -> dict[str, Any]:
    used = {isa.find_bit(q).index for inst in isa.data for q in inst.qubits}
    try:
        duration = float(isa.estimate_duration(device.target))
    except Exception:  # noqa: BLE001 - an estimate only; fall back to a typical layer time
        duration = isa.depth() * 0.5e-6
    return {
        "two_qubit_gates": int(isa.num_nonlocal_gates()),
        "depth": int(isa.depth()),
        "qubits_used": len(used),
        "duration_s": duration,
    }


def estimate_qpu_seconds(shots: int, duration_s: float) -> float:
    """A conservative upper-bound estimate of the QPU time one sampling job will bill."""
    return (JOB_OVERHEAD_S + shots * (REP_DELAY_S + duration_s)) * SAFETY_FACTOR


def estimate_postprocessing_seconds(cfg: SqdConfig, norb: int, nelec: tuple[int, int]) -> float:
    """Rough classical time for the recovery and subspace solves (grows fast with orbitals)."""
    strings = min(cfg.samples_per_batch, comb(norb, nelec[0]))
    solves = cfg.iterations * cfg.num_batches
    return 0.5 + 5e-7 * strings**2 * norb**2 * solves


# ------------------------------------------------------------------ plan


@dataclass
class SqdHardwarePlan:
    """What will be submitted and what it should cost. Made without using any QPU time."""

    plan_id: str
    created: float
    cfg: SqdConfig
    backend: str
    qubits: int
    stats: dict[str, Any]
    estimated_qpu_s: float
    estimated_post_s: float
    remaining_qpu_s: float | None
    pending_jobs: int | None
    cap_s: float
    budget_s: float = FREE_PLAN_BUDGET_S
    problems: list[str] = field(default_factory=list)
    # Not serialised: the transpiled circuit, the device and the molecule data it was made from.
    isa: Any = field(default=None, repr=False)
    device: Any = field(default=None, repr=False)
    problem: Any = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        s = self.stats
        return {
            "plan_id": self.plan_id,
            "expires_at": self.created + PLAN_TTL_S,
            "molecule": self.cfg.molecule.name,
            "backend": self.backend,
            "qubits": self.qubits,
            "qubits_used": s["qubits_used"],
            "two_qubit_gates": s["two_qubit_gates"],
            "depth": s["depth"],
            "shots": self.cfg.shots,
            "per_shot_us": (REP_DELAY_S + s["duration_s"]) * 1e6,
            "estimated_qpu_s": self.estimated_qpu_s,
            "estimated_post_s": self.estimated_post_s,
            "remaining_qpu_s": self.remaining_qpu_s,
            "budget_s": self.budget_s,
            "cap_s": self.cap_s,
            "pending_jobs": self.pending_jobs,
            "exact_available": self.problem.e_exact is not None if self.problem else None,
            "n_determinants": n_determinants(self.problem.norb, self.problem.nelec)
            if self.problem
            else None,
            "problems": self.problems,
            "confirm_phrase": CONFIRM_PHRASE,
        }


def prepare_plan(
    cfg: SqdConfig,
    *,
    backend_name: str | None = None,
    service: Any = None,
    problem: SqdProblem | None = None,
    max_qpu_seconds: float = DEFAULT_MAX_QPU_S,
    optimization_level: int = 3,
    now: float | None = None,
) -> SqdHardwarePlan:
    """Pick the device, transpile the circuit for it, and estimate the cost. Submits nothing."""
    problem = problem or build_problem(cfg.molecule, cfg.bond_length)
    if service is None:
        service = get_service()
    device = resolve_backend(service, backend_name, problem.num_qubits)
    isa = transpile_for(build_circuit(problem, cfg.n_reps), device, optimization_level, cfg.seed)
    stats = circuit_stats(isa, device)
    est = estimate_qpu_seconds(cfg.shots, stats["duration_s"])
    remaining = remaining_qpu_seconds(service)
    try:
        pending = int(device.status().pending_jobs)
    except Exception:  # noqa: BLE001 - queue depth is informational
        pending = None
    problems = []
    if est > max_qpu_seconds:
        problems.append(
            f"The estimate of {est:.0f} s of QPU time is above this server's limit of "
            f"{max_qpu_seconds:.0f} s per run. Lower the shots."
        )
    if remaining is not None and est > remaining:
        problems.append(
            f"The estimate of {est:.0f} s is more than the {remaining:.0f} s of QPU time left."
        )
    return SqdHardwarePlan(
        plan_id=uuid.uuid4().hex[:12],
        created=time.time() if now is None else now,
        cfg=cfg,
        backend=device.name,
        qubits=problem.num_qubits,
        stats=stats,
        estimated_qpu_s=est,
        estimated_post_s=estimate_postprocessing_seconds(cfg, problem.norb, problem.nelec),
        remaining_qpu_s=remaining,
        pending_jobs=pending,
        cap_s=max_qpu_seconds,
        problems=problems,
        isa=isa,
        device=device,
        problem=problem,
    )


# ------------------------------------------------------------------ submit and collect


def _default_sampler(device):
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=device)  # job mode, the only mode on the free plan
    sampler.options.dynamical_decoupling.enable = True
    return sampler


def submit(
    plan: SqdHardwarePlan,
    confirm: str,
    *,
    allow_hardware: bool,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    sampler_factory: Callable[[Any], Any] = _default_sampler,
    now: float | None = None,
):
    """Send the planned circuit to the device. Every gate is checked again here, in code."""
    if not allow_hardware:
        raise HardwareDisabledError(
            "Real hardware is disabled. Start with --allow-hardware to enable it."
        )
    if confirm.strip() != CONFIRM_PHRASE:
        raise HardwareAbortedError("The confirmation was not given; nothing was sent.")
    if plan.problems:
        raise ValueError("This plan cannot be submitted: " + " ".join(plan.problems))
    if (time.time() if now is None else now) - plan.created > PLAN_TTL_S:
        raise ValueError("This plan has expired. Make a new plan, then submit it.")
    sampler = sampler_factory(plan.device)
    job = sampler.run([plan.isa], shots=plan.cfg.shots)
    job_id = str(job.job_id())
    record_job(
        hardware_dir,
        job_id,
        {
            "kind": "sqd",
            "backend": plan.backend,
            "shots": plan.cfg.shots,
            "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": plan.cfg.model_dump(mode="json"),
            "estimated_qpu_s": plan.estimated_qpu_s,
            **plan.stats,
        },
    )
    return job


def wait_for(
    job,
    on_status: Callable[[str], None] | None = None,
    poll_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until the IBM job ends, reporting its state; raise if it did not finish cleanly."""
    while True:
        raw = job.status()
        name = str(getattr(raw, "name", raw)).upper()
        if on_status:
            on_status(name)
        if name in FINAL_STATES:
            break
        sleep(poll_s)
    if name != "DONE":
        raise RuntimeError(f"The IBM job ended as {name}; no result was produced.")


def _ideal_reference(
    cfg: SqdConfig, problem: SqdProblem, rng: np.random.Generator
) -> dict[str, Any] | None:
    """The same device circuit on an ideal simulator, when small enough to simulate."""
    if n_determinants(problem.norb, problem.nelec) > MAX_DETERMINANTS:
        return None
    matrix, probs = sample_lucj(
        problem, cfg.n_reps, cfg.shots, 0.0, rng, device_interaction_pairs(problem.norb)
    )
    ref = diagonalise_samples(cfg, problem, matrix, probs, rng)
    return {
        "error_mha": ref.abs_error_mha,
        "below_hf_mha": ref.below_hf_mha,
        "distinct_samples": ref.n_distinct_samples,
    }


def result_from_counts(
    cfg: SqdConfig,
    problem: SqdProblem,
    counts: dict[str, int],
    *,
    backend: str,
    job_id: str,
    info: dict[str, Any],
) -> SqdResult:
    """Run the classical half of SQD on bitstrings that came from a real device."""
    start = time.perf_counter()
    matrix, probs = counts_to_arrays(counts)
    rng = np.random.default_rng(cfg.seed)
    result = diagonalise_samples(cfg, problem, matrix, probs, rng)
    ideal = _ideal_reference(cfg, problem, np.random.default_rng(cfg.seed))
    result.hardware = {
        "backend": backend,
        "job_id": job_id,
        "circuit": "LUCJ with device-friendly couplings",
        "ideal_same_circuit": ideal,
        **info,
    }
    result.config_hash = hashlib.sha256(
        f"{cfg.config_hash()}|{backend}|{job_id}".encode()
    ).hexdigest()[:12]
    result.library_versions = library_versions()
    result.git_commit = git_commit()
    result.timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    result.wall_time_s = time.perf_counter() - start
    return result


def collect(
    job, plan: SqdHardwarePlan, on_status: Callable[[str], None] | None = None
) -> SqdResult:
    """Wait for a submitted job, then turn its bitstrings into a saved-ready result."""
    wait_for(job, on_status)
    counts = job.result()[0].data.meas.get_counts()
    try:
        qpu_s = float(job.usage())
    except Exception:  # noqa: BLE001 - usage is informational
        qpu_s = None
    info = {
        **plan.stats,
        "shots": plan.cfg.shots,
        "estimated_qpu_s": plan.estimated_qpu_s,
        "qpu_seconds": qpu_s,
    }
    return result_from_counts(
        plan.cfg, plan.problem, counts, backend=plan.backend, job_id=str(job.job_id()), info=info
    )


def finalize_saved_job(
    job_id: str,
    *,
    hardware_dir: Path = DEFAULT_HARDWARE_DIR,
    service: Any = None,
    on_status: Callable[[str], None] | None = None,
) -> SqdResult:
    """Rebuild the result of an earlier submission from its saved record (uses no QPU time)."""
    path = Path(hardware_dir) / "jobs" / f"{job_id}.json"
    record = json.loads(path.read_text())
    if record.get("kind") != "sqd":
        raise ValueError(f"Job {job_id} is not an SQD job.")
    cfg = SqdConfig.model_validate(record["config"])
    service = service or get_service()
    job = service.job(job_id)
    wait_for(job, on_status)
    counts = job.result()[0].data.meas.get_counts()
    problem = build_problem(cfg.molecule, cfg.bond_length)
    try:
        qpu_s = float(job.usage())
    except Exception:  # noqa: BLE001
        qpu_s = None
    info = {
        k: record[k]
        for k in ("two_qubit_gates", "depth", "qubits_used", "duration_s", "estimated_qpu_s")
        if k in record
    }
    info.update(shots=cfg.shots, qpu_seconds=qpu_s)
    return result_from_counts(
        cfg, problem, counts, backend=record["backend"], job_id=job_id, info=info
    )
