"""The trust report: for one recommended setup, the evidence behind it and a plain verdict.

A recommendation is only useful if you can tell how much to rely on it. For each one this lists what
was checked (how many seeds, how often it reached chemical accuracy, whether the exact answer was
known, whether another device agrees, whether the error forecast matched, whether a real device was
tried) and turns the list into one verdict by fixed rules that are returned with the report, so
nothing about the verdict is hidden.

Every function takes plain dicts (the same shapes the API already returns), so the rules can be
tested without a server.
"""

from __future__ import annotations

import re

CHEMICAL_ACCURACY_MHA = 1.6
MIN_SEEDS = 3  # fewer than this and the verdict is "not enough evidence"
GOOD_SEEDS = 5  # this many or more counts as a solid repeat
ACCURATE_SHARE = 0.5  # share of seeds within chemical accuracy that counts as reliable

PASS, WARN, FAIL, NONE = "pass", "warn", "fail", "none"

VERDICT_TEXT = {
    "trusted": "Trusted",
    "partly": "Partly trusted",
    "not_yet": "Not yet trusted",
    "no_evidence": "Not enough evidence",
}

RULES = [
    "Not enough evidence: fewer than 3 seeds, so nothing can be said about reliability.",
    "Not yet trusted: it does not reach chemical accuracy in at least half of its seeds.",
    "Partly trusted: it is reliably accurate, but at least one other check is a warning "
    "or a failure.",
    "Trusted: it is reliably accurate and no check is a warning or a failure. A check with no "
    "data does not count against it, and is shown as such.",
]


def item(id_: str, title: str, status: str, detail: str, route: str | None = None) -> dict:
    return {"id": id_, "title": title, "status": status, "detail": detail, "route": route}


def decide(evidence: list[dict], n_seeds: int, accurate: bool) -> tuple[str, str]:
    """Verdict key and one sentence for it, from the evidence list."""
    if n_seeds < MIN_SEEDS:
        return "no_evidence", f"Only {n_seeds} seed(s): too few to say how reliable this is."
    if not accurate:
        return "not_yet", "It does not reach chemical accuracy in at least half of its seeds."
    problems = [e["title"] for e in evidence if e["status"] in (WARN, FAIL)]
    if problems:
        return "partly", "Reliably accurate in simulation, but: " + "; ".join(problems) + "."
    return "trusted", "Reliably accurate, and every check that has data passes."


# ---------------------------------------------------------------- VQE


def _norm(label: str) -> str:
    """A study label without its '[150 evals, 4096 shots]' budget suffix."""
    return re.sub(r"\s*\[.*\]\s*$", "", label).strip()


def _parts(label: str) -> tuple[str, str, str, str, str] | None:
    """(mapping, ansatz, reps, optimizer, mitigation) from 'parity / hea_hf(1) / spsa / readout'."""
    bits = [b.strip() for b in _norm(label).split("/")]
    if len(bits) != 4:
        return None
    m = re.match(r"(\w+)\((\d+)\)", bits[1])
    if not m:
        return None
    return bits[0], m.group(1), m.group(2), bits[2], bits[3]


def _hardware_parts(label: str) -> tuple[str, str, str, str, str] | None:
    """The same five parts from a hardware record's label: 'hea(reps=2)/spsa, parity, readout'."""
    m = re.match(r"(\w+)\(reps=(\d+)\)\s*/\s*(\w+),\s*(\w+),\s*(\S+)$", label.strip())
    if not m:
        return None
    ansatz, reps, optimizer, mapping, mitigation = m.groups()
    return mapping, ansatz, reps, optimizer, mitigation


def vqe_trust(
    study: dict,
    device_blocks: list[dict],
    hardware: list[dict],
    forecast: dict | None,
) -> dict:
    """Trust report for the recommended configuration of one VQE study."""
    rec = study.get("recommendation")
    if not rec:
        return _empty(study, "There is no recommendation for this study yet.")
    config = next((c for c in study["configs"] if c["label"] == rec["label"]), None)
    if config is None:
        return _empty(study, "The recommended configuration is missing from the study.")
    n = config["n_seeds"]
    k = config["n_chem"]
    evidence = []

    if n >= GOOD_SEEDS:
        evidence.append(
            item("seeds", "Repeated over several seeds", PASS, f"{n} seeds.", "reliability")
        )
    elif n >= MIN_SEEDS:
        evidence.append(
            item(
                "seeds",
                "Repeated over only a few seeds",
                WARN,
                f"{n} seeds; {GOOD_SEEDS} or more is solid.",
                "reliability",
            )
        )
    else:
        evidence.append(item("seeds", "Too few seeds", FAIL, f"{n} seed(s).", "reliability"))

    accurate = n > 0 and k / n >= ACCURATE_SHARE
    detail = (
        f"{k} of {n} seeds within {CHEMICAL_ACCURACY_MHA} mHa "
        f"(95% interval {100 * config['p_chem_lo']:.0f}% to {100 * config['p_chem_hi']:.0f}%); "
        f"mean error {config['mean']:.2f} ± {config['std']:.2f} mHa."
    )
    evidence.append(
        item(
            "accuracy",
            "Reaches chemical accuracy reliably"
            if accurate
            else "Does not reach chemical accuracy reliably",
            PASS if accurate else FAIL,
            detail,
            "reliability",
        )
    )

    evidence.append(
        item(
            "exact",
            "Scored against the exact energy",
            PASS,
            "Every error is the distance from the exact energy of this small molecule.",
            None,
        )
    )

    evidence.append(_devices_item(study, rec["label"], device_blocks))
    evidence.append(_forecast_item(config, forecast))
    evidence.append(_vqe_hardware_item(study, rec["label"], hardware, config))

    key, sentence = decide(evidence, n, accurate)
    return {
        "kind": "vqe",
        "subject": {
            "molecule": study["molecule"],
            "backend": study["backend"],
            "device": study["device"],
            "bond_length": study["bond_length"],
            "label": rec["label"],
        },
        "verdict": key,
        "verdict_text": VERDICT_TEXT[key],
        "sentence": sentence,
        "evidence": evidence,
        "rules": RULES,
    }


def _empty(study: dict, why: str) -> dict:
    return {
        "kind": "vqe",
        "subject": {
            "molecule": study["molecule"],
            "backend": study["backend"],
            "device": study["device"],
            "bond_length": study["bond_length"],
            "label": None,
        },
        "verdict": "no_evidence",
        "verdict_text": VERDICT_TEXT["no_evidence"],
        "sentence": why,
        "evidence": [],
        "rules": RULES,
    }


def _devices_item(study: dict, label: str, blocks: list[dict]) -> dict:
    title = "Holds across devices"
    block = next((b for b in blocks if b["molecule"] == study["molecule"]), None)
    if block is None or study["backend"] != "noisy" or len(block["devices"]) < 2:
        return item(
            "devices", title, NONE, "No multi-device comparison for this molecule.", "devices"
        )
    target = _norm(label)
    same = [d for d, r in block["recommendations"].items() if _norm(r["label"]) == target]
    total = len(block["devices"])
    if len(same) == total:
        return item(
            "devices",
            title,
            PASS,
            f"The same setup is recommended on all {total} devices.",
            "devices",
        )
    if same:
        return item(
            "devices",
            "Holds on only some devices",
            WARN,
            f"The same setup is recommended on {len(same)} of {total} devices; "
            "the others prefer a different one.",
            "devices",
        )
    return item(
        "devices",
        "Not the pick on other devices",
        WARN,
        f"None of the {total} compared devices recommends this exact setup.",
        "devices",
    )


def _forecast_item(config: dict, forecast: dict | None) -> dict:
    title = "Matches the error forecast"
    if not forecast:
        return item(
            "forecast",
            title,
            NONE,
            "No forecast available (ideal backend, or no trained predictor).",
            "predictor",
        )
    low, high, mid = forecast["low_mha"], forecast["high_mha"], forecast["error_mha"]
    measured = config["mean"]
    if low <= measured <= high:
        return item(
            "forecast",
            title,
            PASS,
            f"Forecast {mid:.2f} mHa (80% range {low:.2f} to {high:.2f}); "
            f"measured mean {measured:.2f} mHa is inside it.",
            "predictor",
        )
    return item(
        "forecast",
        "Outside the error forecast",
        WARN,
        f"Forecast {mid:.2f} mHa (80% range {low:.2f} to {high:.2f}); "
        f"measured mean {measured:.2f} mHa is outside it.",
        "predictor",
    )


def _vqe_hardware_item(study: dict, label: str, hardware: list[dict], config: dict) -> dict:
    title = "Checked on a real device"
    mine = _parts(label)
    same_molecule = [h for h in hardware if h["molecule"] == study["molecule"]]
    if not same_molecule:
        return item("hardware", title, NONE, "No real-hardware run for this molecule.", "hardware")
    match = next((h for h in same_molecule if _hardware_parts(h["source_label"]) == mine), None)
    if match is None:
        return item(
            "hardware",
            "Not checked on a real device",
            WARN,
            f"There are real-hardware runs for {study['molecule']}, but none of this exact setup.",
            "hardware",
        )
    modes = {e["mitigation"]: e["error_mha"] for e in match["evaluations"]}
    own = modes.get(config["mitigation"], min(modes.values()))
    hf = match["hf_error_mha"]
    where = (
        f"{match['device']}: {own:.2f} mHa with {config['mitigation']} mitigation "
        f"(simulator predicted {match['simulator_error_mha']:.2f})."
    )
    if own <= CHEMICAL_ACCURACY_MHA:
        return item("hardware", title, PASS, where, "hardware")
    if own < hf:
        return item(
            "hardware",
            "Real device: better than Hartree-Fock, not chemically accurate",
            WARN,
            where,
            "hardware",
        )
    return item("hardware", "Real device: no better than Hartree-Fock", FAIL, where, "hardware")


# ---------------------------------------------------------------- SQD

_DEFAULT_SETTING = {
    "n_reps": 1,
    "readout_noise": 0.02,
    "iterations": 3,
    "samples_per_batch": 300,
    "num_batches": 3,
    "shots": 20000,
}


def _is_default_setting(g: dict) -> bool:
    return (
        all(g.get(k) == v for k, v in _DEFAULT_SETTING.items())
        and g.get("noise_model") is None
        and g.get("sampler", "circuit") == "circuit"
    )


def sqd_trust(
    molecule: str,
    groups: list[dict],
    rows: list[dict],
    controls: list[dict],
    default_bonds: dict[str, float],
) -> dict:
    """Trust report for SQD on one molecule at its equilibrium bond length."""
    bond = default_bonds.get(molecule)
    here = [
        g
        for g in groups
        if g["molecule"] == molecule and (bond is None or abs(g["bond_length"] - bond) < 1e-6)
    ]
    main = next((g for g in here if _is_default_setting(g)), None)
    subject = {"molecule": molecule, "bond_length": bond, "label": "SQD, default setting"}
    if main is None:
        return {
            "kind": "sqd",
            "subject": subject,
            "verdict": "no_evidence",
            "verdict_text": VERDICT_TEXT["no_evidence"],
            "sentence": "No seed study of the default setting for this molecule.",
            "evidence": [],
            "rules": RULES,
        }
    n = main["n_seeds"]
    share = main["share_chem_acc"]
    evidence = []
    if n >= GOOD_SEEDS:
        evidence.append(item("seeds", "Repeated over several seeds", PASS, f"{n} seeds.", "sqd"))
    elif n >= MIN_SEEDS:
        evidence.append(
            item(
                "seeds",
                "Repeated over only a few seeds",
                WARN,
                f"{n} seeds; {GOOD_SEEDS} or more is solid.",
                "sqd",
            )
        )
    else:
        evidence.append(item("seeds", "Too few seeds", FAIL, f"{n} seed(s).", "sqd"))

    accurate = share >= ACCURATE_SHARE
    evidence.append(
        item(
            "accuracy",
            "Reaches chemical accuracy reliably"
            if accurate
            else "Does not reach chemical accuracy reliably",
            PASS if accurate else FAIL,
            f"{round(share * n)} of {n} seeds within {CHEMICAL_ACCURACY_MHA} mHa; "
            f"mean error {main['mean_error_mha']:.2f} ± {main['std_error_mha']:.2f} mHa.",
            "sqd",
        )
    )

    ccsd = main.get("ccsd_error_mha")
    if ccsd is None:
        evidence.append(
            item(
                "ccsd",
                "Compared with classical CCSD",
                NONE,
                "No exact energy at this size, so no error to compare.",
                "sqd",
            )
        )
    elif main["mean_error_mha"] < ccsd:
        evidence.append(
            item(
                "ccsd",
                "Closer to exact than classical CCSD",
                PASS,
                f"SQD {main['mean_error_mha']:.2f} mHa against CCSD {ccsd:.2f} mHa.",
                "sqd",
            )
        )
    else:
        evidence.append(
            item(
                "ccsd",
                "Not closer than classical CCSD",
                WARN,
                f"SQD {main['mean_error_mha']:.2f} mHa against CCSD {ccsd:.2f} mHa.",
                "sqd",
            )
        )

    evidence.append(_sqd_control_item(molecule, here, controls))
    evidence.append(_sqd_stretch_item(molecule, groups, bond))
    evidence.append(_sqd_hardware_item(molecule, rows))

    key, sentence = decide(evidence, n, accurate)
    return {
        "kind": "sqd",
        "subject": subject,
        "verdict": key,
        "verdict_text": VERDICT_TEXT[key],
        "sentence": sentence,
        "evidence": evidence,
        "rules": RULES,
    }


def _mean(v: list[float]) -> float:
    return sum(v) / len(v)


def _sqd_control_item(molecule: str, groups: list[dict], controls: list[dict]) -> dict:
    """Does the circuit beat classical ways of choosing the subspace, and the loop without it?

    Two kinds of control are used. Subspaces of the same size picked without a circuit are the
    fair test. The loop run with random bits or a noisy Hartree-Fock state in place of the circuit
    shows whether the circuit is needed at all; it solves a different share of the space, so it is
    judged separately.
    """
    title = "The circuit adds value over controls"
    control = next((c for c in controls if c["molecule"] == molecule), None)
    if control is None:
        return item("controls", title, NONE, "No control experiments for this molecule.", "sqd")
    one = [s for s in control["sqd"] if s["n_reps"] == 1]
    if not one:
        return item(
            "controls", title, NONE, "No equal-size subspace comparison for this molecule.", "sqd"
        )
    circuit = _mean([s["error_mha"] for s in one])  # the same seeds the baselines were run for
    same_size = {
        "a random subspace": _mean([s["random_mean_mha"] for s in one]),
        "a low-excitation subspace": _mean([s["excitation_mha"] for s in one]),
    }
    swapped = {}
    for sampler, name in (
        ("uniform", "the loop with random bits"),
        ("hf_noise", "the loop with a noisy Hartree-Fock state"),
    ):
        g = next(
            (
                g
                for g in groups
                if g["molecule"] == molecule
                and g.get("sampler") == sampler
                and g["n_reps"] == 1
                and g["iterations"] == 3
                and g["samples_per_batch"] == 300
            ),
            None,
        )
        if g is not None:
            swapped[name] = g["mean_error_mha"]
    lost = [n for n, v in {**same_size, **swapped}.items() if circuit >= v]
    parts = [f"circuit {circuit:.2f} mHa"]
    parts += [f"{n} {v:.2f}" for n, v in {**same_size, **swapped}.items()]
    detail = "At the first seeds: " + "; ".join(parts) + "."
    if not lost:
        return item(
            "controls", title, PASS, detail + " The circuit is lower than every control.", "sqd"
        )
    return item(
        "controls",
        "The circuit does not beat every control",
        WARN,
        detail + f" Not lower than: {', '.join(lost)}.",
        "sqd",
    )


def _sqd_stretch_item(molecule: str, groups: list[dict], bond: float | None) -> dict:
    title = "Holds when bonds stretch"
    study = [
        g
        for g in groups
        if g["molecule"] == molecule and g["n_reps"] == 1 and _is_default_setting(g)
    ]
    bonds = sorted({g["bond_length"] for g in study})
    if len(bonds) < 3:
        return item("stretch", title, NONE, "No stretched-bond study for this molecule.", "sqd")
    pairs = [
        (g["mean_error_mha"], g["ccsd_error_mha"])
        for g in study
        if g.get("ccsd_error_mha") is not None
    ]
    closer = sum(1 for s, c in pairs if s < c)
    detail = (
        f"Closer to the exact energy than CCSD at {closer} of {len(pairs)} bond lengths "
        f"({bonds[0]} to {bonds[-1]} Å)."
    )
    if closer * 2 >= len(pairs):
        return item("stretch", title, PASS, detail, "sqd")
    return item("stretch", "Does not hold up when bonds stretch", WARN, detail, "sqd")


def _sqd_hardware_item(molecule: str, rows: list[dict]) -> dict:
    title = "Checked on a real device"
    hw = [
        r for r in rows if r["molecule"] == molecule and r.get("hardware") and r["status"] == "ok"
    ]
    if not hw:
        return item("hardware", title, NONE, "No real-hardware SQD run for this molecule.", "sqd")
    errs = [r["error_mha"] for r in hw if r["error_mha"] is not None]
    fractions = [r["subspace_fraction"] for r in hw if r["subspace_fraction"] is not None]
    devices = sorted({r["hardware"]["backend"] for r in hw})
    detail = f"{len(hw)} run(s) on {', '.join(devices)}."
    if errs:
        detail += f" Errors {min(errs):.3f} to {max(errs):.3f} mHa."
    if fractions and min(fractions) > 0.5:
        return item(
            "hardware",
            "Ran on a real device, but it solved most of the space",
            WARN,
            detail
            + f" Each solved {100 * min(fractions):.0f}% or more of the full space, so this shows "
            "the hardware path works, not that SQD tolerates noise at scale.",
            "sqd",
        )
    return item("hardware", title, PASS, detail, "sqd")
