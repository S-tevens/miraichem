"""The MiraiChem web API: browse results, rank and recommend, and launch simulator runs and sweeps.

Everything here wraps the same tested package the CLI uses. Runs and sweeps are simulator work
(ideal and noisy backends). The one real-hardware path is SQD sampling: it needs a plan with a
cost estimate and a confirmation, and ``miraichem serve --no-allow-hardware`` switches it off.
VQE hardware runs stay in the command line (``miraichem hardware``).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from miraichem.api.data import OPTIONS, Workspace, ranked_row, reference_energies, run_row
from miraichem.api.jobs import JobManager
from miraichem.api.sqd import (
    load_controls,
    load_rows,
    sqd_default_bonds,
    sqd_groups,
    sqd_molecules,
)
from miraichem.api.views import clean, recommendation_json
from miraichem.backends.hardware import get_service
from miraichem.benchmark.ranking import Weights, recommend
from miraichem.benchmark.ranking_replicated import recommend_replicated
from miraichem.benchmark.ranking_scan import recommend_across_scan
from miraichem.benchmark.sweep import load_sweep_config, unique_runs
from miraichem.config import MoleculeConfig, RunConfig
from miraichem.predict import ErrorPredictor, nearest_runs
from miraichem.predict.service import shape_for_config
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.config import load_sqd_yaml
from miraichem.sqd.hardware import (
    CONFIRM_PHRASE,
    DEFAULT_MAX_QPU_S,
    PLAN_TTL_S,
    SqdHardwarePlan,
    collect,
    prepare_plan,
)
from miraichem.sqd.hardware import _default_sampler as default_sampler
from miraichem.sqd.hardware import submit as submit_to_hardware
from miraichem.sqd.problem import SqdProblem, build_problem
from miraichem.sqd.storage import save_sqd_result


class RunRequest(BaseModel):
    """A single VQE run to start. Only simulator backends are accepted."""

    molecule: str = "h2"
    bond_length: float = Field(0.735, gt=0, le=10)
    mapping: Literal["jordan_wigner", "parity"] = "parity"
    ansatz: Literal["uccsd", "hea", "hea_hf"] = "uccsd"
    ansatz_reps: int = Field(1, ge=1, le=6)
    optimizer: Literal["cobyla", "spsa", "spsa_clip", "lbfgsb"] = "cobyla"
    maxiter: int = Field(150, ge=1, le=5000)
    backend: Literal["ideal", "noisy"] = "ideal"
    fake_backend_name: str = "FakeSherbrooke"
    shots: int = Field(4096, ge=1, le=1_000_000)
    mitigation: Literal["none", "readout", "zne", "readout+zne"] = "none"
    seed: int = 42
    force: bool = False


class SqdRequest(BaseModel):
    """A single SQD run to start (sampling is simulated, so this never uses a quantum device)."""

    molecule: str = "n2"
    bond_length: float | None = Field(None, gt=0, le=10, description="empty = the molecule's own")
    n_reps: int = Field(1, ge=1, le=4)
    shots: int = Field(20000, ge=100, le=200_000)
    readout_noise: float = Field(0.02, ge=0, lt=0.5)
    noise_model: str | None = Field(None, description="fake device name; replaces readout_noise")
    exact: bool = Field(False, description="also solve the exact energy (can take minutes)")
    iterations: int = Field(3, ge=1, le=8)
    samples_per_batch: int = Field(300, ge=10, le=5000)
    num_batches: int = Field(3, ge=1, le=10)
    seed: int = 42


class SqdHardwareRequest(BaseModel):
    """Plan a real-device SQD run. Planning transpiles for the device and uses no QPU time."""

    molecule: str = "n2"
    bond_length: float | None = Field(None, gt=0, le=10)
    backend: str | None = Field(None, description="device name; empty = least busy")
    n_reps: int = Field(1, ge=1, le=4)
    shots: int = Field(10000, ge=100, le=100_000)
    iterations: int = Field(3, ge=1, le=8)
    samples_per_batch: int = Field(300, ge=10, le=5000)
    num_batches: int = Field(3, ge=1, le=10)
    seed: int = 42


class SqdHardwareRun(BaseModel):
    plan_id: str
    confirm: str


class SweepRequest(BaseModel):
    max_workers: int = Field(4, ge=1, le=16)
    force: bool = False


def _run_config(ws: Workspace, req: RunRequest) -> RunConfig:
    """Build and validate the RunConfig for a request, turning problems into HTTP errors."""
    try:
        mol = MoleculeConfig(**yaml.safe_load(ws.molecule_yaml(req.molecule)))
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Unknown molecule {req.molecule!r}.") from exc
    try:
        return RunConfig(
            molecule=mol,
            bond_length=req.bond_length,
            mapping=req.mapping,
            ansatz=req.ansatz,
            ansatz_reps=req.ansatz_reps,
            optimizer=req.optimizer,
            maxiter=req.maxiter,
            backend=req.backend,
            fake_backend_name=req.fake_backend_name,
            shots=req.shots,
            mitigation=req.mitigation,
            seed=req.seed,
        )
    except ValidationError as exc:
        msg = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        raise HTTPException(422, msg) from exc


def _source(ws: Workspace, source: str | None) -> str:
    s = source or ws.default_source()
    if s not in ("local", "published"):
        raise HTTPException(404, f"Unknown data source {s!r}.")
    return s


def _comment_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(x for x in lines if x)


def create_app(
    results_dir: Path = Path("results"),
    published_dir: Path = Path("results/published"),
    configs_dir: Path = Path("configs"),
    static_dir: Path | None = None,
    *,
    allow_hardware: bool = False,
    max_qpu_seconds: float = DEFAULT_MAX_QPU_S,
    hardware_service=None,
    sampler_factory=None,
) -> FastAPI:
    ws = Workspace(results_dir, published_dir, configs_dir)
    jobs = JobManager(Path(results_dir), on_change=lambda: ws.invalidate("local"))
    # Read the saved runs once in the background so the first page load is not the slow one.
    threading.Thread(
        target=lambda: [ws.results(src) for src in ("published", "local")], daemon=True
    ).start()
    app = FastAPI(
        title="MiraiChem", version="1.0", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )

    # ------------------------------------------------------------------ error predictor

    model_name = Path("models") / "error_predictor.json"
    loaded: dict = {"path": None, "mtime": 0.0, "model": None}

    def current_model() -> tuple[ErrorPredictor | None, str | None]:
        """The newest trained predictor: your own if you trained one, else the published one."""
        for label, root in (("local", Path(results_dir)), ("published", Path(published_dir))):
            path = root / model_name
            if path.is_file():
                mtime = path.stat().st_mtime
                if loaded["path"] != str(path) or loaded["mtime"] != mtime:
                    loaded.update(path=str(path), mtime=mtime, model=ErrorPredictor.load(path))
                return loaded["model"], label
        return None, None

    @app.get("/api/predictor")
    def predictor_status():
        model, origin = current_model()
        n_now = sum(
            1 for r in ws.results("local") if r.status == "ok" and r.config.backend == "noisy"
        )
        if model is None:
            return {"available": False, "noisy_runs_available": n_now}
        return clean(
            {
                "available": True,
                "origin": origin,
                "report": model.report,
                "noisy_runs_available": n_now,
                "newer_data": n_now > model.report["n_runs"],
            }
        )

    @app.post("/api/predictor/train", status_code=202)
    def train_predictor():
        def work(job) -> None:
            job.message = "Training the predictor (about a minute)..."
            model = ErrorPredictor.train(
                ws.results("local"), progress=lambda m: setattr(job, "message", m + "...")
            )
            model.save(Path(results_dir) / model_name)
            job.message = "Trained and saved."

        return jobs.submit_task("train", "Train the error predictor", work).as_dict()

    @app.post("/api/predict")
    def predict(req: RunRequest):
        if req.backend != "noisy":
            return {
                "supported": False,
                "reason": "Predictions are for the noisy simulator: "
                "an ideal run has no device noise to forecast.",
            }
        model, origin = current_model()
        if model is None:
            raise HTTPException(
                404, "No trained predictor. Train one with `miraichem predictor train`."
            )
        cfg = _run_config(ws, req)
        try:
            shape = shape_for_config(cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"Could not prepare the circuit: {exc}") from exc
        pred = model.predict(shape)
        source = "local" if ws.has_runs("local") else "published"
        similar = nearest_runs(ws.results(source), shape)
        return clean(
            {
                "supported": True,
                "origin": origin,
                "prediction": pred.as_dict(),
                "circuit": {
                    "two_qubit_gates": shape.two_qubit_gates,
                    "depth": shape.depth,
                    "qubits": shape.qubits,
                    "parameters": shape.parameters,
                },
                "similar": [run_row(r) for r in similar],
            }
        )

    # ------------------------------------------------------------------ SQD

    @app.get("/api/structures")
    def structures():
        """3D structures and orbital surfaces for the molecule viewer (made by make_structures)."""
        path = Path(published_dir) / "structures" / "structures.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []

    @app.get("/api/sqd/options")
    def sqd_options():
        return {"molecules": sqd_molecules(configs_dir, Path(results_dir) / "exact")}

    @app.get("/api/sqd")
    def sqd_results(source: str | None = None):
        s = _source(ws, source)
        rows = load_rows(ws.path(s))
        return clean(
            {
                "source": s,
                "runs": rows,
                "groups": sqd_groups(rows),
                "default_bonds": sqd_default_bonds(configs_dir),
                "controls": load_controls(ws.path(s)),
            }
        )

    @app.post("/api/sqd/run", status_code=202)
    def start_sqd(req: SqdRequest):
        path = Path(configs_dir) / "sqd" / f"{Path(req.molecule).stem}.yaml"
        if not path.is_file():
            raise HTTPException(404, f"Unknown SQD molecule {req.molecule!r}.")
        molecules = sqd_molecules(configs_dir, Path(results_dir) / "exact")
        size = next((m for m in molecules if m["id"] == path.stem), None)
        if size is not None and not size["simulable"]:
            raise HTTPException(
                422,
                f"{size['name']} has {size['qubits']} qubits: too large to simulate here. "
                "It can only be sampled on a real device.",
            )
        mol, default_bond = load_sqd_yaml(path.read_text())
        try:
            cfg = SqdConfig(
                molecule=mol,
                bond_length=req.bond_length or default_bond,
                n_reps=req.n_reps,
                shots=req.shots,
                readout_noise=0.0 if req.noise_model else req.readout_noise,
                noise_model=req.noise_model or None,
                iterations=req.iterations,
                samples_per_batch=req.samples_per_batch,
                num_batches=req.num_batches,
                seed=req.seed,
            )
        except ValidationError as exc:
            msg = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
            raise HTTPException(422, msg) from exc

        def work(job) -> None:
            job.message = "Solving the classical references and sampling the circuit..."
            if req.exact:
                job.message = "Solving the exact energy once (it is saved, so this is a one-off)..."
            result = run_sqd(
                cfg, exact=True if req.exact else None, exact_cache=Path(results_dir) / "exact"
            )
            save_sqd_result(result, Path(results_dir))
            job.result_hashes = [result.config_hash]
            if result.status == "failed":
                raise RuntimeError(result.error_message)
            job.best_error_mha = result.abs_error_mha
            job.message = "Finished."

        title = f"SQD, {cfg.molecule.name}, noise {cfg.readout_noise:g}, seed {cfg.seed}"
        return jobs.submit_task("sqd", title, work).as_dict()

    # ---- real hardware: plan (no QPU time), then submit behind a typed confirmation

    hardware_dir = Path(results_dir) / "hardware"
    plans: dict[str, SqdHardwarePlan] = {}
    submitted: set[str] = set()
    problem_cache: dict[tuple, SqdProblem] = {}
    plan_lock = threading.Lock()
    service_factory = hardware_service or get_service
    make_sampler = sampler_factory or default_sampler

    def sqd_problem(molecule_cfg: MoleculeConfig, bond_length: float) -> SqdProblem:
        key = (molecule_cfg.model_dump_json(), round(bond_length, 6))
        with plan_lock:
            if key not in problem_cache:
                if len(problem_cache) >= 3:
                    problem_cache.pop(next(iter(problem_cache)))
                problem_cache[key] = build_problem(
                    molecule_cfg, bond_length, cache_dir=Path(results_dir) / "exact"
                )
            return problem_cache[key]

    def require_hardware() -> None:
        if not allow_hardware:
            raise HTTPException(
                403,
                "Real hardware is switched off on this server. Restart the app with "
                "`miraichem serve --allow-hardware` to enable it.",
            )

    @app.get("/api/sqd/hardware")
    def sqd_hardware_status():
        from dotenv import load_dotenv

        load_dotenv()
        return {
            "enabled": allow_hardware,
            "token_configured": bool(os.getenv("IBM_QUANTUM_TOKEN")),
            "max_qpu_seconds": max_qpu_seconds,
            "confirm_phrase": CONFIRM_PHRASE,
            "plan_valid_seconds": PLAN_TTL_S,
        }

    @app.post("/api/sqd/hardware/plan", status_code=202)
    def plan_sqd_hardware(req: SqdHardwareRequest):
        require_hardware()
        path = Path(configs_dir) / "sqd" / f"{Path(req.molecule).stem}.yaml"
        if not path.is_file():
            raise HTTPException(404, f"Unknown SQD molecule {req.molecule!r}.")
        mol, default_bond = load_sqd_yaml(path.read_text())
        bond = req.bond_length or default_bond
        try:
            cfg = SqdConfig(
                molecule=mol,
                bond_length=bond,
                n_reps=req.n_reps,
                shots=req.shots,
                readout_noise=0.0,  # a real device supplies the noise
                iterations=req.iterations,
                samples_per_batch=req.samples_per_batch,
                num_batches=req.num_batches,
                seed=req.seed,
            )
        except ValidationError as exc:
            msg = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
            raise HTTPException(422, msg) from exc

        def work(job) -> None:
            job.message = "Solving the classical references (no QPU time is used)..."
            problem = sqd_problem(mol, bond)
            job.message = "Choosing the device and transpiling the circuit for it..."
            plan = prepare_plan(
                cfg,
                backend_name=(req.backend or "").strip() or None,
                service=service_factory(),
                problem=problem,
                max_qpu_seconds=max_qpu_seconds,
            )
            plans[plan.plan_id] = plan
            job.result_hashes = [plan.plan_id]
            job.message = "Plan ready. Nothing has been submitted."

        title = f"Plan hardware SQD, {mol.name}, {req.shots} shots"
        return jobs.submit_task("sqd-plan", title, work).as_dict()

    @app.get("/api/sqd/hardware/plans/{plan_id}")
    def sqd_hardware_plan(plan_id: str):
        require_hardware()
        plan = plans.get(plan_id)
        if plan is None:
            raise HTTPException(404, "Unknown plan. Make a new plan.")
        out = plan.as_dict()
        out["expired"] = time.time() > out["expires_at"]
        out["submitted"] = plan_id in submitted
        return clean(out)

    @app.post("/api/sqd/hardware/run", status_code=202)
    def run_sqd_hardware(req: SqdHardwareRun):
        require_hardware()
        plan = plans.get(req.plan_id)
        if plan is None:
            raise HTTPException(404, "Unknown plan. Make a new plan.")
        if req.confirm.strip() != CONFIRM_PHRASE:
            raise HTTPException(422, f"Type {CONFIRM_PHRASE!r} exactly to confirm.")
        if plan.problems:
            raise HTTPException(422, "This plan cannot be submitted: " + " ".join(plan.problems))
        if time.time() - plan.created > PLAN_TTL_S:
            raise HTTPException(410, "This plan has expired. Make a new plan.")
        with plan_lock:
            if plan.plan_id in submitted:
                raise HTTPException(409, "This plan was already submitted.")
            submitted.add(plan.plan_id)

        def work(job) -> None:
            job.message = f"Submitting to {plan.backend}..."
            try:
                ibm_job = submit_to_hardware(
                    plan,
                    req.confirm,
                    allow_hardware=allow_hardware,
                    hardware_dir=hardware_dir,
                    sampler_factory=make_sampler,
                )
            except Exception:
                submitted.discard(plan.plan_id)  # nothing reached the device, so it may be retried
                raise
            ibm_id = str(ibm_job.job_id())
            job.details["ibm_job_id"] = ibm_id
            job.details["backend"] = plan.backend
            job.message = f"Submitted as {ibm_id}. Waiting for {plan.backend}..."
            result = collect(
                ibm_job,
                plan,
                on_status=lambda s: setattr(job, "message", f"IBM job {ibm_id}: {s.lower()}"),
            )
            save_sqd_result(result, Path(results_dir))
            job.result_hashes = [result.config_hash]
            job.best_error_mha = result.abs_error_mha
            job.message = "Finished."

        title = f"Hardware SQD, {plan.cfg.molecule.name}, {plan.backend}"
        return jobs.submit_task("sqd-hardware", title, work).as_dict()

    # ------------------------------------------------------------------ meta

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/sources")
    def sources():
        return {"sources": ws.sources(), "default": ws.default_source()}

    @app.get("/api/options")
    def options():
        return {**OPTIONS, "molecules": ws.molecules()}

    @app.get("/api/overview")
    def overview(source: str | None = None):
        s = _source(ws, source)
        results = ws.results(s)
        ok = [r for r in results if r.status == "ok"]
        by: dict[str, dict[str, int]] = {}
        for r in ok:
            by.setdefault(r.config.molecule.name, {}).setdefault(r.config.backend, 0)
            by[r.config.molecule.name][r.config.backend] += 1
        best = sorted(
            (r for r in ok if r.abs_error_mha is not None and r.config.backend == "noisy"),
            key=lambda r: r.abs_error_mha or 0,
        )[:1]
        return clean(
            {
                "source": s,
                "n_runs": len(results),
                "n_ok": len(ok),
                "n_failed": len(results) - len(ok),
                "by_molecule": by,
                "hardware_results": ws.hardware_available(s),
                "curves": len(ws.curves(s)),
                "has_replication": bool(ws.studies(s)),
                "best_noisy": run_row(best[0]) if best else None,
                "hf_error_mha": ws.hf_errors(s),
                "recent": [
                    run_row(r)
                    for r in sorted(ok, key=lambda r: r.timestamp or "", reverse=True)[:8]
                ],
            }
        )

    # ------------------------------------------------------------------ results

    @app.get("/api/runs")
    def runs(source: str | None = None, molecule: str | None = None, backend: str | None = None):
        s = _source(ws, source)
        rows = [
            run_row(r)
            for r in ws.results(s)
            if (not molecule or r.config.molecule.name.lower() == molecule.lower())
            and (not backend or r.config.backend == backend)
        ]
        return clean({"source": s, "runs": rows})

    @app.get("/api/runs/{run_hash}")
    def run_detail(run_hash: str, source: str | None = None):
        s = _source(ws, source)
        r = ws.find_run(s, run_hash)
        if r is None:
            raise HTTPException(404, f"Run {run_hash} not found in {s}.")
        mol_id = r.config.molecule.name.lower()
        cmd = (
            f"miraichem run --molecule configs/molecules/{mol_id}.yaml "
            f"--bond-length {r.config.bond_length} --mapping {r.config.mapping} "
            f"--ansatz {r.config.ansatz} --ansatz-reps {r.config.ansatz_reps} "
            f"--optimizer {r.config.optimizer} --maxiter {r.config.maxiter} "
            f"--backend {r.config.backend} --shots {r.config.shots} "
            f"--mitigation {r.config.mitigation} --seed {r.config.seed}"
        )
        return clean({"run": r.model_dump(mode="json"), "row": run_row(r), "command": cmd})

    @app.get("/api/rank")
    def rank(
        molecule: str,
        backend: str = "noisy",
        bond_length: float | None = None,
        w_acc: float = 0.7,
        w_gates: float = 0.2,
        w_shots: float = 0.1,
        source: str | None = None,
    ):
        s = _source(ws, source)
        try:
            weights = Weights(w_acc, w_gates, w_shots)
            rec = recommend(ws.results(s), molecule, backend, weights, bond_length)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        bonds = sorted(
            {
                r.config.bond_length
                for r in ws.results(s)
                if r.status == "ok"
                and r.config.molecule.name.lower() == molecule.lower()
                and r.config.backend == backend
            }
        )
        return clean(
            {
                "source": s,
                "bond_length": rec.best.result.config.bond_length,
                "bond_lengths": bonds,
                "ranked": [ranked_row(x) for x in rec.ranked],
                "best_hash": rec.best.result.config_hash,
                "justification": rec.justification,
                "reached_chemical_accuracy": rec.reached_chemical_accuracy,
            }
        )

    @app.get("/api/scan")
    def scan(
        molecule: str,
        backend: str = "ideal",
        w_acc: float = 0.7,
        w_gates: float = 0.2,
        w_shots: float = 0.1,
        source: str | None = None,
    ):
        s = _source(ws, source)
        try:
            rec = recommend_across_scan(
                ws.results(s), molecule, backend, Weights(w_acc, w_gates, w_shots)
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return clean(
            {
                "source": s,
                "justification": rec.justification,
                "best_key": rec.best.key,
                "everywhere": rec.reached_chemical_accuracy_everywhere,
                "ranked": [
                    {
                        "key": x.key,
                        "rank": x.rank,
                        "label": x.label,
                        "hash": x.result.config_hash,
                        "points": x.n_ok,
                        "attempted": x.n_attempted,
                        "worst_error_mha": x.worst_error_mha,
                        "mean_error_mha": x.mean_error_mha,
                        "coverage": x.coverage,
                        "gates": x.gates,
                        "shots": x.shots,
                        "score": x.score,
                    }
                    for x in rec.ranked
                ],
            }
        )

    @app.get("/api/replicated")
    def replicated(source: str | None = None):
        s = _source(ws, source)
        return {"source": s, "studies": ws.studies(s)}

    @app.get("/api/replicated/recommendation")
    def replicated_recommendation(
        molecule: str,
        backend: str = "noisy",
        device: str | None = None,
        reliability: float = Query(0.5, ge=0, le=1),
        alpha: float = Query(0.05, gt=0, lt=1),
        margin_factor: float = Query(1.5, ge=1),
        source: str | None = None,
    ):
        s = _source(ws, source)
        try:
            rec = recommend_replicated(
                ws.replication_runs(s),
                molecule,
                backend,
                min_seeds=3,
                reliability=reliability,
                alpha=alpha,
                margin_factor=margin_factor,
                device=device or None,
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return clean(recommendation_json(rec))

    @app.get("/api/devices")
    def devices(source: str | None = None):
        s = _source(ws, source)
        return {"source": s, "blocks": ws.devices(s)}

    @app.get("/api/hardware")
    def hardware(source: str | None = None):
        s = _source(ws, source)
        return {"source": s, "results": ws.hardware(s)}

    @app.get("/api/curves")
    def curves(source: str | None = None):
        s = _source(ws, source)
        return {"source": s, "curves": ws.curves(s)}

    # ------------------------------------------------------------------ launching work

    @app.get("/api/reference")
    def reference(molecule: str, bond_length: float):
        try:
            return reference_energies(ws.molecule_yaml(molecule), round(bond_length, 6))
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Unknown molecule {molecule!r}.") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"Could not compute the reference: {exc}") from exc

    @app.post("/api/runs", status_code=202)
    def start_run(req: RunRequest):
        cfg = _run_config(ws, req)
        mol = cfg.molecule
        title = f"{mol.name} · {cfg.ansatz}/{cfg.optimizer} · {cfg.backend} · {cfg.bond_length} Å"
        return jobs.submit_run(cfg, title, force=req.force).as_dict()

    @app.get("/api/sweeps")
    def list_sweeps():
        out = []
        saved = ws.ok_hashes("local")
        for p in sorted((ws.configs_dir / "sweeps").glob("*.yaml")):
            try:
                cfg = load_sweep_config(p)
                planned = unique_runs(cfg.expand())
            except Exception as exc:  # noqa: BLE001
                out.append({"name": p.stem, "error": str(exc)})
                continue
            n_cached = sum(r.config_hash() in saved for r in planned)
            out.append(
                {
                    "name": p.stem,
                    "description": _comment_block(p.read_text()),
                    "molecule": cfg.molecule.name,
                    "n_runs": len(planned),
                    "n_cached": n_cached,
                    "backends": sorted({r.backend for r in planned}),
                    "ansatzes": sorted({r.ansatz for r in planned}),
                    "optimizers": sorted({r.optimizer for r in planned}),
                    "seeds": len({r.seed for r in planned}),
                    "devices": sorted(
                        {r.fake_backend_name for r in planned if r.backend != "ideal"}
                    ),
                }
            )
        return {"sweeps": out, "busy": jobs.sweep_running()}

    @app.post("/api/sweeps/{name}/start", status_code=202)
    def start_sweep(name: str, req: SweepRequest):
        try:
            path = ws.sweep_path(name)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Unknown sweep {name!r}.") from exc
        if jobs.sweep_running():
            raise HTTPException(409, "A sweep is already running. Wait for it to finish.")
        cfg = load_sweep_config(path)
        return jobs.submit_sweep(cfg, f"Sweep · {path.stem}", req.max_workers, req.force).as_dict()

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.as_dict() for j in jobs.list()]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown job.")
        return j.as_dict()

    # ------------------------------------------------------------------ the built frontend

    if static_dir and (Path(static_dir) / "index.html").is_file():
        static = Path(static_dir)
        if (static / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found.")
            candidate = (static / path).resolve()
            if path and candidate.is_file() and static.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static / "index.html")

    return app
