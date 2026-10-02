import numpy as np
import pytest

from miraichem.optim.optimizers import initial_point, minimize


def quad(x: np.ndarray) -> float:
    return float(np.sum((x - 0.3) ** 2))


@pytest.mark.parametrize("kind", ["cobyla", "lbfgsb", "spsa"])
def test_optimizers_reduce_quadratic(kind):
    x0 = np.zeros(3)
    res = minimize(kind, quad, x0, maxiter=300, seed=1)
    assert res.fun < quad(x0)
    assert res.fun < 0.05
    assert res.nfev == len(res.trace)
    assert res.nfev <= 310  # budget respected (small slack for finite-difference gradients)


def test_spsa_is_seeded():
    a = minimize("spsa", quad, np.zeros(3), maxiter=60, seed=5)
    b = minimize("spsa", quad, np.zeros(3), maxiter=60, seed=5)
    c = minimize("spsa", quad, np.zeros(3), maxiter=60, seed=6)
    assert a.trace == b.trace
    assert a.trace != c.trace


def test_initial_point_small_and_reproducible():
    x = initial_point(8, seed=3)
    assert np.allclose(x, initial_point(8, seed=3))
    assert np.max(np.abs(x)) < 0.1
