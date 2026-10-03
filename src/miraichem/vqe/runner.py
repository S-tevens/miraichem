"""The VQE loop: build the problem, optimise the ansatz parameters, record everything.

VQE (variational quantum eigensolver) guesses a ground state with a parameterised circuit,
measures its energy <H> on a quantum backend, and lets a classical optimiser tweak the parameters
to lower that energy. By the variational principle the energy can never go below the true ground
state, so lower is better and the exact energy is the floor.
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from miraichem.ansatz import build_ansatz
from miraichem.backends.base import EnergyBackend
from miraichem.backends.ideal import IdealBackend
from miraichem.backends.noisy import NoisyBackend
from miraichem.benchmark.metrics import abs_error_mha, within_chemical_accuracy
from miraichem.chemistry.classical import compute_reference
from miraichem.config import RunConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.mitigation.mitigated import MitigatedBackend
from miraichem.optim.optimizers import initial_point, minimize
from miraichem.vqe.result import RunResult

log = logging.getLogger(__name__)

_LIBS = ["qiskit", "qiskit-aer", "qiskit-ibm-runtime", "qiskit-nature", "pyscf", "numpy", "scipy"]


def library_versions() -> dict[str, str]:
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = "not installed"
    return out


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def make_backend(cfg: RunConfig) -> EnergyBackend:
    """Create the energy backend for a run config."""
    if cfg.backend == "ideal":
        return IdealBackend()
    if cfg.backend == "noisy":
        noisy = NoisyBackend(cfg.fake_backend_name, cfg.shots, cfg.seed)
        if cfg.mitigation == "none":
            return noisy
        return MitigatedBackend(noisy, cfg.mitigation, cfg.zne_scales, cfg.zne_method)
    raise NotImplementedError(f"Backend {cfg.backend!r} is not implemented yet.")


def run_vqe(cfg: RunConfig, backend: EnergyBackend | None = None) -> RunResult:
    """Run one VQE configuration. Never raises: failures come back as ``status='failed'``."""
    base = RunResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        library_versions=library_versions(),
        git_commit=git_commit(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    start = time.perf_counter()
    try:
        return _run(cfg, base, backend or make_backend(cfg), start)
    except Exception as exc:  # noqa: BLE001 - a failed run must not stop a sweep
        log.exception("Run %s failed", base.config_hash)
        base.status = "failed"
        base.error_message = f"{type(exc).__name__}: {exc}"
        base.wall_time_s = time.perf_counter() - start
        return base


def _run(cfg: RunConfig, result: RunResult, backend: EnergyBackend, start: float) -> RunResult:
    ref = compute_reference(cfg.molecule, cfg.bond_length)
    problem = build_qubit_hamiltonian(cfg.molecule, cfg.bond_length, cfg.mapping)
    ansatz = build_ansatz(cfg.ansatz, problem, cfg.ansatz_reps)
    circuit, hamiltonian = ansatz.circuit, problem.hamiltonian

    unmitigated: list[float] = []

    def energy(theta: np.ndarray) -> float:
        # Total energy = <H_qubit> + constant shift (nuclear repulsion, frozen core).
        est = backend.estimate(circuit, hamiltonian, theta)
        if "unmitigated" in est.metadata:
            unmitigated.append(est.metadata["unmitigated"] + problem.energy_shift)
        return est.value + problem.energy_shift

    x0 = initial_point(ansatz.num_parameters, cfg.seed)
    opt = minimize(cfg.optimizer, energy, x0, cfg.maxiter, cfg.seed)
    e_final = energy(opt.x)  # fresh evaluation at the optimum (the reported VQE energy)

    result.e_hf, result.e_exact = ref.e_hf, ref.e_exact
    result.e_vqe = e_final
    result.abs_error_mha = abs_error_mha(e_final, ref.e_exact)
    result.within_chemical_accuracy = within_chemical_accuracy(e_final, ref.e_exact)
    result.num_qubits = problem.num_qubits
    result.num_params = ansatz.num_parameters
    result.logical_depth = ansatz.logical_depth
    result.n_function_evals = opt.nfev + 1
    result.total_shots = backend.total_shots
    result.convergence_trace = opt.trace + [e_final]
    if unmitigated:
        result.convergence_trace_unmitigated = unmitigated
        result.e_vqe_unmitigated = unmitigated[-1]  # raw energy at the final (reported) evaluation
    result.optimal_parameters = [float(v) for v in opt.x]
    result.backend_info = backend.info()
    if isinstance(backend, NoisyBackend | MitigatedBackend):
        metrics = backend.transpiled_metrics(circuit, hamiltonian)
        result.transpiled_depth = metrics["transpiled_depth"]
        result.two_qubit_gates = metrics["two_qubit_gates"]
    result.wall_time_s = time.perf_counter() - start
    return result
