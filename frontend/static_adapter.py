"""Translate a native optimization record for the dependency-free frontend."""

from __future__ import annotations

from frontend.molecule3d import molecule_payload
from frontend.science import (
    candidate_reviews,
    chosen_candidate,
    describe_change,
    recommended_assays,
)


def _structure(smiles: str, parent: str | None = None) -> dict | None:
    try:
        return molecule_payload(smiles, parent)
    except (RuntimeError, ValueError):
        return None


def _outcome(parent: dict, candidate: dict) -> list[str]:
    deltas = {
        axis: candidate["normalised"][axis] - parent["normalised"][axis]
        for axis in candidate["normalised"]
    }
    improved = sorted(
        (axis for axis, value in deltas.items() if value > 0.005),
        key=deltas.get,
        reverse=True,
    )
    reduced = sorted(
        (axis for axis, value in deltas.items() if value < -0.005),
        key=deltas.get,
    )
    notes = []
    if improved:
        notes.append(
            "Improved "
            + ", ".join(f"{axis} ({deltas[axis]:+.2f})" for axis in improved)
            + "."
        )
    if reduced:
        notes.append(
            "Reduced "
            + ", ".join(f"{axis} ({deltas[axis]:+.2f})" for axis in reduced)
            + "."
        )
    return notes


def to_static_run(record: dict, compound_name: str | None = None) -> dict:
    """Return the JSON contract consumed by ``static/app.js``."""
    program = record.get("program") or {}
    name = compound_name or program.get("compound") or "Starting compound"
    seed = record["history"][0]
    rounds = []
    all_candidates = []

    for round_record in record["round_records"]:
        round_number = round_record["round"]
        chosen = chosen_candidate(round_record)
        candidates = round_record["candidates"]
        candidate_ids = {
            candidate["smiles"]: f"r{round_number}-c{index}"
            for index, candidate in enumerate(candidates, 1)
        }
        display_names = {
            smiles: f"{name} analogue R{round_number}.{index}"
            for index, smiles in enumerate(candidate_ids, 1)
        }
        review_rows = candidate_reviews(round_record, program)
        confidence = {review["agent"]: review for review in review_rows}

        transcript = []
        for review in round_record["reviews"]:
            metadata = confidence.get(review["agent"], {})
            transcript.append(
                {
                    "agent": review["agent"],
                    "verdict": review["verdict"],
                    "reason": review["reason"],
                    "confidence": metadata.get("confidence"),
                    "confidence_basis": metadata.get("confidence_basis", ""),
                }
            )
        orchestrator = confidence.get("Orchestrator", {})
        transcript.append(
            {
                "agent": "orchestrator",
                "verdict": "selected",
                "rationale": round_record["decision"]["rationale"],
                "confidence": orchestrator.get("confidence"),
                "confidence_basis": orchestrator.get("confidence_basis", ""),
            }
        )

        static_reviews = []
        for review in review_rows:
            static_reviews.append(
                {
                    "agent": review["agent"],
                    "confidence": review["confidence"],
                    "confidence_basis": review["confidence_basis"],
                    "candidates": [
                        {
                            "id": candidate_ids.get(row["smiles"], ""),
                            "display_name": display_names.get(
                                row["smiles"], row["label"]
                            ),
                            "summary": row["score"],
                            "disposition": row["disposition"].lower(),
                            "change": row["change"],
                            "reason": (
                                f"{row['reason']} {row['target_context']}"
                            ).strip(),
                        }
                        for row in review["candidates"]
                    ],
                }
            )

        assays = [
            {
                "priority": assay["priority"],
                "assay": assay["name"],
                "purpose": assay["why"],
                "readout": assay["readout"],
            }
            for assay in recommended_assays(round_record, program)
        ]
        winner_id = candidate_ids.get(
            chosen["smiles"], f"r{round_number}-incumbent"
        )
        winner = {
            "id": winner_id,
            "display_name": display_names.get(
                chosen["smiles"], f"{name} incumbent R{round_number}"
            ),
            "smiles": chosen["smiles"],
            "scores": chosen["normalised"],
        }
        static_candidates = [
            {
                "id": candidate_ids[candidate["smiles"]],
                "display_name": display_names[candidate["smiles"]],
                "smiles": candidate["smiles"],
                "scores": candidate["normalised"],
            }
            for candidate in candidates
        ]
        all_candidates.extend(static_candidates)
        rounds.append(
            {
                "round": round_number,
                "parent": round_record["parent"]["smiles"],
                "candidates": static_candidates,
                "transcript": transcript,
                "winner": winner,
                "changes": {
                    "structural": [describe_change(round_record, chosen)],
                    "outcome": _outcome(round_record["parent"], chosen),
                },
                "recommended_assays": assays,
                "agent_candidate_review": static_reviews,
                "structure_3d": _structure(
                    chosen["smiles"], round_record["parent"]["smiles"]
                ),
                "previous_scores": round_record["parent"]["normalised"],
            }
        )

    provenance = record.get("provenance") or {}
    if provenance.get("any_surrogate"):
        notice = (
            "At least one labelled surrogate is active. Use the displayed "
            "predictions for decision support and verify them experimentally."
        )
    else:
        notice = "Bundled ADMET models are active; experimental verification remains required."
    return {
        "schema_version": "1.0",
        "mode": record.get("mode", "specialists"),
        "target": record["target"],
        "seed_name": name,
        "seed": record["seed"],
        "seed_structure_3d": _structure(record["seed"]),
        "seed_scores": seed["normalised"],
        "target_rationale": program.get("rationale", ""),
        "optimization_goal": program.get("objective", ""),
        "hard_constraint": (
            "The safety specialist applies a binding hERG veto; the "
            "orchestrator cannot restore a vetoed candidate."
        ),
        "oracle_notice": notice,
        "rounds": rounds,
        "all_candidates": all_candidates,
        "final": rounds[-1]["winner"] if rounds else None,
    }
