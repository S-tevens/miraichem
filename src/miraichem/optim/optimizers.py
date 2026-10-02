"""Uniform optimizer interface over COBYLA, SPSA and L-BFGS-B.

``maxiter`` means the same thing for every optimizer: the budget of objective-function
EVALUATIONS (each evaluation is one energy estimate, i.e. the expensive quantum step). This keeps
comparisons fair, since SPSA uses 2 evaluations per iteration and COBYLA uses 1.
Every evaluation is recorded so convergence plots show the true cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from miraichem.config import OptimizerKind

Objective = Callable[[np.ndarray], float]


@dataclass
class OptResult:
    x: np.ndarray
    fun: float  # objective at the best-scoring evaluated point
    nfev: int
    trace: list[float] = field(
        default_factory=list
    )  # objective value of every evaluation, in order


def initial_point(num_parameters: int, seed: int, scale: float = 0.01) -> np.ndarray:
    """Small random start near zero (zero = Hartree-Fock for UCCSD). Avoids poor random starts
    and barren plateaus; the seed makes runs reproducible."""
    return np.random.default_rng(seed).normal(0.0, scale, num_parameters)


class _Recorder:
    """Wraps the objective to count and record every evaluation."""

    def __init__(self, fun: Objective) -> None:
        self.fun = fun
        self.trace: list[float] = []

    def __call__(self, x: np.ndarray) -> float:
        value = float(self.fun(np.asarray(x, dtype=float)))
        self.trace.append(value)
        return value


def _spsa(rec: _Recorder, x0: np.ndarray, max_iter: int, seed: int) -> np.ndarray:
    """Simultaneous Perturbation Stochastic Approximation (Spall).

    Estimates the gradient from just 2 evaluations per step by perturbing ALL parameters at once
    with random +/-1 signs, which makes it cheap and tolerant to shot noise.
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    c0, alpha, gamma = 0.1, 0.602, 0.101
    stability = 0.1 * max_iter

    # Calibrate the step size so the first update moves each parameter by about 0.1.
    grads = []
    for _ in range(3):
        delta = rng.choice([-1.0, 1.0], n)
        grads.append(abs(rec(x0 + c0 * delta) - rec(x0 - c0 * delta)) / (2 * c0))
    a0 = 0.1 * (stability + 1) ** alpha / max(float(np.mean(grads)), 1e-8)

    x = x0.copy()
    for k in range(max_iter):
        a_k = a0 / (k + 1 + stability) ** alpha
        c_k = c0 / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], n)
        g = (rec(x + c_k * delta) - rec(x - c_k * delta)) / (2 * c_k) * (1.0 / delta)
        x = x - a_k * g
    return x


def minimize(
    kind: OptimizerKind, fun: Objective, x0: np.ndarray, maxiter: int, seed: int = 0
) -> OptResult:
    """Minimise ``fun`` from ``x0`` with at most about ``maxiter`` function evaluations."""
    rec = _Recorder(fun)
    x0 = np.asarray(x0, dtype=float)
    if kind == "cobyla":
        res = scipy_minimize(rec, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
        x = res.x
    elif kind == "lbfgsb":
        # Finite-difference gradients cost n+1 evaluations each; only valid on exact backends.
        res = scipy_minimize(rec, x0, method="L-BFGS-B", options={"maxfun": maxiter, "ftol": 1e-12})
        x = res.x
    elif kind == "spsa":
        budget = max((maxiter - 6) // 2, 1)  # 6 evaluations go to calibration
        x = _spsa(rec, x0, budget, seed)
    else:
        raise ValueError(f"Unknown optimizer: {kind!r}")
    best = int(np.argmin(rec.trace))
    return OptResult(x=np.asarray(x), fun=rec.trace[best], nfev=len(rec.trace), trace=rec.trace)
