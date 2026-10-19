"""Predict the error of a noisy VQE run before running it, with honest uncertainty.

Model: a random forest on pre-run features (see ``features``) predicting log10(error in mHa).
Everything that is reported is validated on configurations the model has not seen:

- 5-fold cross-validation grouped by configuration (all seeds of a configuration stay together);
- baselines (the global mean, and the per-molecule mean) on the same folds;
- leave-one-device-out, a harder test: predict a device that was not in the training data;
- uncertainty from the cross-validated residuals, calibrated on other folds only, so the stated
  80% interval can be checked against how often it really contained the answer;
- permutation importance, to say what the model relies on.

The model is only as good as the data it saw: it knows H2 on five fake IBM devices and LiH on one,
and :meth:`ErrorPredictor.predict` warns when a request is outside that.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from miraichem.benchmark.metrics import CHEMICAL_ACCURACY_HA
from miraichem.predict.features import (
    FEATURES,
    RunShape,
    group_key,
    shape_of,
    target_of,
    vector,
)
from miraichem.predict.forest import RandomForest
from miraichem.vqe.result import RunResult

LOG_CA = math.log10(CHEMICAL_ACCURACY_HA * 1000)
N_FOLDS = 5
INTERVAL = (0.1, 0.9)  # an 80% interval
MIN_ROWS = 60


@dataclass
class Prediction:
    log_error: float
    error_mha: float
    low_mha: float
    high_mha: float
    p_chemical_accuracy: float
    p_beats_hartree_fock: float
    hf_error_mha: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _folds(groups: list[str], k: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    unique = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(unique))
    assign = {unique[i]: pos % k for pos, i in enumerate(order)}
    return np.array([assign[g] for g in groups])


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae_log10": float(np.abs(err).mean()),
        "rmse_log10": float(np.sqrt((err**2).mean())),
        "typical_factor": float(10 ** np.abs(err).mean()),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else 0.0,
        "n": int(len(y)),
    }


class ErrorPredictor:
    """A trained predictor plus everything needed to explain how good it is."""

    def __init__(
        self,
        forest: RandomForest,
        residuals: dict[str, list[float]],
        report: dict[str, Any],
        domain: dict[str, Any],
    ) -> None:
        self.forest = forest
        self.residuals = residuals  # signed out-of-fold residuals (actual - predicted), by molecule
        self.report = report
        self.domain = domain

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        results: list[RunResult],
        progress: Callable[[str], None] | None = None,
        n_trees: int = 60,
    ) -> ErrorPredictor:
        say = progress or (lambda _m: None)
        rows = [(shape_of(r), r) for r in results]
        rows = [(s, r) for s, r in rows if s is not None]
        if len(rows) < MIN_ROWS:
            raise ValueError(
                f"Need at least {MIN_ROWS} finished noisy runs to train; found {len(rows)}."
            )
        shapes = [s for s, _ in rows]
        X = np.array([vector(s) for s in shapes])
        y = np.array([target_of(r) for _, r in rows])
        mol = np.array([s.config.molecule.name for s in shapes])
        dev = np.array([s.config.fake_backend_name for s in shapes])
        groups = [group_key(s.config) for s in shapes]
        folds = _folds(groups)

        def fit(Xt: np.ndarray, yt: np.ndarray) -> RandomForest:
            return RandomForest(n_trees=n_trees).fit(Xt, yt)

        say("Cross-validating on unseen configurations")
        oof = np.zeros(len(y))
        fold_models: list[RandomForest] = []
        for k in range(N_FOLDS):
            train, test = folds != k, folds == k
            m = fit(X[train], y[train])
            fold_models.append(m)
            oof[test] = m.predict(X[test])

        # baselines on the same folds
        base_mean = np.zeros(len(y))
        base_mol = np.zeros(len(y))
        for k in range(N_FOLDS):
            train, test = folds != k, folds == k
            base_mean[test] = y[train].mean()
            per_mol = {m_: y[train & (mol == m_)].mean() for m_ in set(mol[train])}
            base_mol[test] = [per_mol.get(m_, y[train].mean()) for m_ in mol[test]]

        # interval calibration done fold by fold: the quantiles for fold k use only OTHER folds
        resid = y - oof
        covered = np.zeros(len(y), dtype=bool)
        for k in range(N_FOLDS):
            test = folds == k
            lo, hi = np.quantile(resid[~test], INTERVAL)
            covered[test] = (resid[test] >= lo) & (resid[test] <= hi)

        say("Holding out each device in turn")
        holdout = []
        for d in sorted(set(dev)):
            test = dev == d
            if test.sum() < 40 or (~test).sum() < MIN_ROWS:
                continue
            m = fit(X[~test], y[~test])
            holdout.append(
                {
                    "device": d,
                    "train_rows": int((~test).sum()),
                    # holding out the device that supplies most of the data leaves too little,
                    # too narrow a training set to be a fair test of generalisation
                    "fair": bool((~test).sum() >= 0.5 * len(y)),
                    **_metrics(m.predict(X[test]), y[test]),
                }
            )

        say("Measuring which inputs the model relies on")
        rng = np.random.default_rng(1)
        base_mae = float(np.abs(oof - y).mean())
        importance = {}
        for j, name in enumerate(FEATURES):
            increase = []
            for k in range(N_FOLDS):
                test = np.where(folds == k)[0]
                Xp = X[test].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                increase.append(
                    float(np.abs(fold_models[k].predict(Xp) - y[test]).mean())
                    - float(np.abs(oof[test] - y[test]).mean())
                )
            importance[name] = float(np.mean(increase))

        say("Fitting the final model")
        final = fit(X, y)

        residuals: dict[str, list[float]] = {"*": [float(v) for v in resid]}
        for m_ in sorted(set(mol)):
            sel = mol == m_
            if sel.sum() >= 100:
                residuals[m_] = [float(v) for v in resid[sel]]

        sample = np.random.default_rng(2).permutation(len(y))[:600]
        report = {
            "trained": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "n_runs": int(len(y)),
            "n_configurations": len(set(groups)),
            "molecules": sorted(set(mol)),
            "devices": sorted(set(dev)),
            "cross_validation": _metrics(oof, y),
            "baseline_mean": _metrics(base_mean, y),
            "baseline_per_molecule": _metrics(base_mol, y),
            "per_molecule": {m_: _metrics(oof[mol == m_], y[mol == m_]) for m_ in sorted(set(mol))},
            "interval": {
                "level": INTERVAL[1] - INTERVAL[0],
                "coverage": float(covered.mean()),
                "median_width_factor": float(
                    10
                    ** np.median(np.quantile(resid, INTERVAL[1]) - np.quantile(resid, INTERVAL[0]))
                ),
            },
            "device_holdout": holdout,
            "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
            "base_mae_log10": base_mae,
            "points": [
                {
                    "predicted": float(oof[i]),
                    "actual": float(y[i]),
                    "molecule": str(mol[i]),
                    "device": str(dev[i]),
                }
                for i in sample
            ],
        }
        domain = {
            "molecules": sorted(set(mol)),
            "devices": sorted(set(dev)),
            "min": [float(v) for v in X.min(axis=0)],
            "max": [float(v) for v in X.max(axis=0)],
            "mitigations": sorted({s.config.mitigation for s in shapes}),
            "optimizers": sorted({s.config.optimizer for s in shapes}),
        }
        return cls(final, residuals, report, domain)

    # ------------------------------------------------------------------ prediction

    def predict(self, shape: RunShape) -> Prediction:
        x = np.array([vector(shape)])
        mu = float(self.forest.predict(x)[0])
        mol = shape.config.molecule.name
        res = np.array(self.residuals.get(mol) or self.residuals["*"])
        lo, hi = np.quantile(res, INTERVAL)
        hf_log = math.log10(max(shape.hf_error_mha, 1e-3))
        return Prediction(
            log_error=mu,
            error_mha=10**mu,
            low_mha=10 ** (mu + float(lo)),
            high_mha=10 ** (mu + float(hi)),
            p_chemical_accuracy=float(np.mean(mu + res < LOG_CA)),
            p_beats_hartree_fock=float(np.mean(mu + res < hf_log)),
            hf_error_mha=shape.hf_error_mha,
            warnings=self._warnings(shape, x[0]),
        )

    def _warnings(self, shape: RunShape, x: np.ndarray) -> list[str]:
        d = self.domain
        c = shape.config
        out = []
        if c.molecule.name not in d["molecules"]:
            out.append(
                f"{c.molecule.name} was not in the training data "
                f"(it saw {', '.join(d['molecules'])}), so this is an extrapolation."
            )
        if c.fake_backend_name not in d["devices"]:
            out.append(f"The device {c.fake_backend_name} was not in the training data.")
        if c.mitigation not in d["mitigations"] or c.optimizer not in d["optimizers"]:
            out.append("This optimizer or mitigation setting was not in the training data.")
        for name, v, lo, hi in zip(FEATURES, x, d["min"], d["max"], strict=True):
            if name in (
                "log_two_qubit_gates",
                "log_depth",
                "qubits",
                "log_budget",
                "log2_shots",
            ) and (v < lo - 1e-9 or v > hi + 1e-9):
                out.append("The circuit size, budget or shots are outside the range the model saw.")
                break
        return out

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "forest": self.forest.to_dict(),
            "residuals": self.residuals,
            "report": self.report,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ErrorPredictor:
        return cls(RandomForest.from_dict(d["forest"]), d["residuals"], d["report"], d["domain"])

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ErrorPredictor:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def nearest_runs(results: list[RunResult], shape: RunShape, limit: int = 5) -> list[RunResult]:
    """Measured runs most like this one (same molecule and device), for a reality check."""
    c = shape.config
    scored: dict[str, tuple[int, float, RunResult]] = {}
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        if (
            r.status == "ok"
            and r.config.backend == "noisy"
            and r.config.molecule.name == c.molecule.name
            and r.config.fake_backend_name == c.fake_backend_name
            and abs(r.config.bond_length - c.bond_length) < 1e-6
        ):
            groups[group_key(r.config)].append(r)
    for key, rs in groups.items():
        rc = rs[0].config
        distance = (
            (rc.ansatz != c.ansatz)
            + (rc.ansatz_reps != c.ansatz_reps)
            + (rc.optimizer != c.optimizer)
            + (rc.mitigation != c.mitigation)
            + (rc.mapping != c.mapping)
            + (rc.maxiter != c.maxiter) * 0.5
            + (rc.shots != c.shots) * 0.5
        )
        mean = float(np.mean([r.abs_error_mha or 0 for r in rs]))
        scored[key] = (int(distance * 10), mean, sorted(rs, key=lambda r: r.config.seed)[0])
    best = sorted(scored.values(), key=lambda t: (t[0], t[1]))[:limit]
    return [r for _, _, r in best]
