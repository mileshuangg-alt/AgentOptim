"""Explanations and experimental suggestions derived from a round record."""

from __future__ import annotations

from core import agents


def chosen_candidate(record: dict) -> dict:
    chosen = record["decision"]["chosen"]
    return next(
        (candidate for candidate in record["candidates"] if candidate["smiles"] == chosen),
        record["parent"],
    )


def describe_change(record: dict, candidate: dict) -> str:
    if candidate["smiles"] == record["parent"]["smiles"]:
        return "No analogue advanced; the orchestrator retained the incumbent."
    transform = candidate.get("transform") or "unlabelled edit"
    intent = candidate.get("intent") or "No edit rationale was recorded."
    return f"{transform.replace('_', ' ').title()}: {intent}."


def recommended_assays(record: dict, program: dict | None = None) -> list[dict[str, str]]:
    program = program or {}
    target = program.get("target") or "DRD2"
    activity_strategy = program.get("activity_strategy") or ""
    bbb_goal = program.get("bbb_goal", "penetrate")
    parent = record["parent"]["normalised"]
    chosen = chosen_candidate(record)["normalised"]
    deltas = {axis: chosen[axis] - parent[axis] for axis in chosen}
    assays = [
        {
            "priority": "Required",
            "name": "LC–MS identity and purity",
            "why": "Confirm the synthesized analogue before interpreting biological data.",
            "readout": "Expected mass, retention profile, and purity.",
        }
    ]
    options = {
        "affinity": {
            "priority": "High",
            "name": f"{target} target-engagement assay",
            "why": (
                "Test target activity directly because structural similarity does "
                "not establish potency."
                if "similarity" in activity_strategy.lower()
                else "Confirm that the edit preserves target engagement."
            ),
            "readout": (
                f"A target-appropriate binding or functional potency curve for {target}."
            ),
        },
        "herg": {
            "priority": "High",
            "name": "Automated hERG patch clamp",
            "why": "Test whether predicted safety reflects lower channel blockade.",
            "readout": "Current inhibition and concentration–response IC50.",
        },
        "bbb": {
            "priority": "Medium",
            "name": "PAMPA-BBB or MDCK-MDR1 permeability",
            "why": (
                "Verify reduced CNS exposure for this peripheral program."
                if bbb_goal == "avoid"
                else "Measure passive CNS penetration and possible efflux liability."
            ),
            "readout": "Apparent permeability and, for MDCK-MDR1, efflux ratio.",
        },
        "solubility": {
            "priority": "Medium",
            "name": "Kinetic and equilibrium solubility",
            "why": "Verify developability under assay-relevant conditions.",
            "readout": "Dissolved concentration by LC–MS across relevant pH conditions.",
        },
        "sa": {
            "priority": "Supporting",
            "name": "Small-scale synthesis feasibility",
            "why": "Test the route and expose unstable intermediates or difficult workups.",
            "readout": "Route length, isolated yield, purity, and handling observations.",
        },
    }
    included = set()
    if "similarity" in activity_strategy.lower():
        assays.append(options["affinity"])
        included.add("affinity")
    for axis in sorted(deltas, key=lambda key: abs(deltas[key]), reverse=True):
        if axis not in included and abs(deltas[axis]) >= 0.03:
            assays.append(options[axis])
            included.add(axis)
        if len(assays) == 4:
            break
    return assays


def candidate_reviews(record: dict, program: dict | None = None) -> list[dict]:
    program = program or {}
    target = program.get("target") or "DRD2"
    objective = program.get("objective") or (
        f"Preserve target activity at {target} while improving developability."
    )
    activity_strategy = program.get("activity_strategy") or ""
    candidates = record["candidates"]
    reviews = {review["agent"]: review for review in record["reviews"]}
    vetoed = set(record["decision"].get("vetoed") or [])
    selected = record["decision"]["chosen"]
    output = []
    definitions = (
        (
            "Affinity",
            lambda c: c["normalised"]["affinity"],
            f"{target} retention",
        ),
        (
            "ADME",
            lambda c: (
                c["normalised"]["solubility"] + c["normalised"]["bbb"]
            )
            / 2,
            "solubility + BBB",
        ),
        ("Safety", lambda c: c["normalised"]["herg"], "hERG safety"),
        ("Orchestrator", lambda c: agents.consensus_score(c["normalised"]), "weighted utility"),
    )
    for name, scorer, score_label in definitions:
        source = record["decision"] if name == "Orchestrator" else reviews.get(name, {})
        preferred = (
            selected if name == "Orchestrator" else source.get("preferred")
        )
        confidence = source.get("confidence")
        confidence_basis = source.get("confidence_basis")
        if confidence is None:
            if name == "Safety":
                line, rule = agents.SAFETY_AGENT.effective_line(record["parent"])
                margin = min(
                    abs(candidate["normalised"]["herg"] - line)
                    for candidate in candidates
                )
                confidence = agents.decision_confidence(margin)
                confidence_basis = (
                    f"The closest candidate is {margin:.3f} normalized hERG units "
                    f"from the binding {rule} veto line."
                )
            elif (
                name == "Orchestrator"
                and preferred == record["parent"]["smiles"]
            ):
                confidence = 0.95
                confidence_basis = (
                    "A binding constraint retained the incumbent for this round."
                )
            else:
                confidence_candidates = (
                    [
                        candidate for candidate in candidates
                        if candidate["smiles"] not in vetoed
                    ]
                    if name == "Orchestrator"
                    else candidates
                )
                metadata = agents.ranked_choice_confidence(
                    confidence_candidates, scorer, preferred, score_label
                )
                confidence = metadata["confidence"]
                confidence_basis = metadata["confidence_basis"]
        rows = []
        best_score = max(scorer(candidate) for candidate in candidates)
        for candidate in sorted(candidates, key=scorer, reverse=True):
            smiles = candidate["smiles"]
            value = scorer(candidate)
            if name == "Safety" and smiles in vetoed:
                disposition = "Vetoed"
                reason = "Rejected by the binding hERG rule for this round."
            elif name == "Orchestrator" and smiles in vetoed:
                disposition = "Ineligible"
                reason = "Safety removed this candidate before final selection."
            elif smiles == preferred:
                disposition = "Selected" if name == "Orchestrator" else "Preferred"
                reason = (
                    record["decision"]["rationale"]
                    if name == "Orchestrator"
                    else reviews.get(name, {}).get("reason", "Highest specialist score.")
                )
            else:
                disposition = "Considered"
                reason = (
                    f"Not advanced by {name}: {score_label} was "
                    f"{best_score - value:.2f} below this column's leading candidate."
                )
            rows.append(
                {
                    "smiles": smiles,
                    "label": candidate.get("transform", "candidate").replace("_", " ").title(),
                    "change": describe_change(record, candidate),
                    "score": f"{score_label}: {value:.2f}",
                    "disposition": disposition,
                    "reason": reason,
                    "target_context": (
                        f"For {target}: {objective} "
                        f"Activity evidence: {activity_strategy}"
                    ).strip(),
                }
            )
        output.append(
            {
                "agent": name,
                "confidence": confidence,
                "confidence_basis": confidence_basis,
                "candidates": rows,
            }
        )
    return output
