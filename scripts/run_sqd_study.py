"""Seed study for SQD on N2 (20 qubits): vary the sampling noise and the SQD settings over seeds.

Usage: python scripts/run_sqd_study.py [results-dir] [--seeds 10]
Results are saved like any other SQD run (results/sqd/n2/<hash>.json) and cached by config hash,
so an interrupted study resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from miraichem.config import MoleculeConfig
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.problem import build_problem
from miraichem.sqd.storage import load_sqd_results, save_sqd_result

ROOT = Path(__file__).resolve().parents[1]

# (label, overrides on the baseline). The baseline is noise 0.02, 3 rounds, 300 samples per batch.
VARIANTS = [
    ("noise 0", {"readout_noise": 0.0}),
    ("noise 0.01", {"readout_noise": 0.01}),
    ("noise 0.02 (baseline)", {}),
    ("noise 0.05", {"readout_noise": 0.05}),
    ("100 samples per batch", {"samples_per_batch": 100}),
    ("1000 samples per batch", {"samples_per_batch": 1000}),
    ("1 round", {"iterations": 1}),
    ("2 circuit layers", {"n_reps": 2}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.results_dir)

    mol = MoleculeConfig(**yaml.safe_load((ROOT / "configs" / "sqd" / "n2.yaml").read_text()))
    problem = build_problem(mol, 1.0977)
    done = {r.config_hash for r in load_sqd_results(out) if r.status == "ok"}
    for (label, extra), seed in itertools.product(VARIANTS, range(args.seeds)):
        base = dict(molecule=mol, bond_length=1.0977, readout_noise=0.02, seed=seed)
        cfg = SqdConfig(**{**base, **extra})
        if cfg.config_hash() in done:
            continue
        res = run_sqd(cfg, problem)
        save_sqd_result(res, out)
        err = "failed" if res.status == "failed" else f"{res.abs_error_mha:.2f} mHa"
        print(f"{label:28s} seed {seed}: {err}", flush=True)


if __name__ == "__main__":
    main()
