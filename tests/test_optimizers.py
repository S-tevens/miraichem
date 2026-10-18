import numpy as np
import pytest

from miraichem.optim.optimizers import initial_point, minimize


def quad(x: np.ndarray) -> float:
    return float(np.sum((x - 0.3) ** 2))


@pytest.mark.parametrize("kind", ["cobyla", "lbfgsb", "spsa", "spsa_clip"])
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


def flat_start_quartic(x: np.ndarray) -> float:
    # Gradient is almost zero at the start (like the Hartree-Fock state) but grows steeply nearby.
    return float(50.0 * np.sum(x**4))


def test_spsa_clip_stays_near_a_flat_start_where_plain_spsa_blows_up():
    x0 = np.full(4, 0.01)
    plain = minimize("spsa", flat_start_quartic, x0, maxiter=60, seed=3)
    clipped = minimize("spsa_clip", flat_start_quartic, x0, maxiter=60, seed=3)
    n_iter = (60 - 6) // 2
    assert np.abs(clipped.x - x0).max() <= 0.05 * n_iter + 1e-9
    assert np.abs(plain.x - x0).max() > 0.05 * n_iter  # the original really does jump
    assert flat_start_quartic(clipped.x) < flat_start_quartic(plain.x)


def test_initial_point_small_and_reproducible():
    x = initial_point(8, seed=3)
    assert np.allclose(x, initial_point(8, seed=3))
    assert np.max(np.abs(x)) < 0.1
