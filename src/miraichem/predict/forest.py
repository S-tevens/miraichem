"""A small random-forest regressor in NumPy (no scikit-learn dependency).

Regression trees are grown on bootstrap samples, each split looking at a random subset of the
features and a handful of quantile thresholds. It is slow-ish compared with a compiled library but
plenty for a few thousand rows, easy to inspect, and serialises to JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Tree = Any  # a float (leaf) or [feature, threshold, left, right]

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _grow(
    X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator, n_try: int, min_leaf: int
) -> Tree:
    n, p = X.shape
    if depth == 0 or n < 2 * min_leaf or float(y.std()) < 1e-9:
        return float(y.mean())
    best: tuple[float, int, float] | None = None
    for j in rng.choice(p, size=min(n_try, p), replace=False):
        col = X[:, j]
        for t in np.unique(np.quantile(col, _QUANTILES)):
            left = col <= t
            n_left = int(left.sum())
            if n_left < min_leaf or n - n_left < min_leaf:
                continue
            sse = float(y[left].var() * n_left + y[~left].var() * (n - n_left))
            if best is None or sse < best[0]:
                best = (sse, int(j), float(t))
    if best is None:
        return float(y.mean())
    _, j, t = best
    left = X[:, j] <= t
    return [
        j,
        t,
        _grow(X[left], y[left], depth - 1, rng, n_try, min_leaf),
        _grow(X[~left], y[~left], depth - 1, rng, n_try, min_leaf),
    ]


def _walk(tree: Tree, x: np.ndarray) -> float:
    while isinstance(tree, list):
        tree = tree[2] if x[tree[0]] <= tree[1] else tree[3]
    return float(tree)


class RandomForest:
    """Bagged regression trees. ``predict`` returns the mean; ``spread`` the std across trees."""

    def __init__(
        self, n_trees: int = 60, max_depth: int = 8, min_leaf: int = 5, seed: int = 0
    ) -> None:
        self.n_trees, self.max_depth, self.min_leaf, self.seed = n_trees, max_depth, min_leaf, seed
        self.trees: list[Tree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n_try = max(3, X.shape[1] // 3)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, len(y), len(y))
            self.trees.append(_grow(X[idx], y[idx], self.max_depth, rng, n_try, self.min_leaf))
        return self

    def _all(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([[_walk(t, x) for t in self.trees] for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._all(X).mean(axis=1)

    def spread(self, X: np.ndarray) -> np.ndarray:
        return self._all(X).std(axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_leaf": self.min_leaf,
            "seed": self.seed,
            "trees": self.trees,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RandomForest:
        rf = cls(d["n_trees"], d["max_depth"], d["min_leaf"], d["seed"])
        rf.trees = d["trees"]
        return rf
