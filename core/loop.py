"""The optimisation loop, and the cached-run format the frontend replays.

One round is: propose from the incumbent, score everything, let the three
specialists argue, apply safety's veto, let the orchestrator choose. The chosen
molecule becomes the next round's incumbent.

`mode="generic"` replaces the three specialists and the orchestrator with a
single all-properties agent and no veto. That is the ablation: same seed, same
oracles, same edit engine, one agent instead of four. It is the only
comparison in this build that is evidence rather than demo.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from core import agents, edits, oracles
from core.contract import AXES, canonical, normalize

HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"

# Axes the demo plots and measures movement on.
#
# NOT affinity. The starting compound is the ceiling of seed-similarity mode,
# and many validated activity oracles also place a known lead near the top.
# That leaves little headroom on the activity axis. Solubility and hERG safety
# remain useful cross-program developability axes, while target retention is
# defended as a constraint.
PARETO_AXES = ("solubility", "herg")

# Target activity or structural retention the optimization is expected to hold.
AFFINITY_FLOOR = 0.60

# How far below the incumbent a candidate may score and still be accepted.
# Strict hill-climbing stalls on a plateau and burns demo rounds re-proposing
# from the same parent; a small tolerance keeps the search moving.
ACCEPT_TOLERANCE = 0.03

# How the next round's proposals are steered. Measured over 12 seeds, 5 rounds,
# surrogate oracles (scripts/ablation.py):
#
#   weakest          0.313 +/- 0.046   attack the incumbent's worst axes
#   none             0.301 +/- 0.035   no steering at all
#   contested-weak   0.274 +/- 0.038   weak axes that an agent also objected about
#   disputed         0.263 +/- 0.031   whatever the agents objected about
#
# The intuitive rule -- steer toward whatever the specialists are arguing about
# -- is the worst of the four, and every rule that consults the agents is worse
# than one that ignores them. An objecting agent names the axis it is defending,
# and steering there proposes edits that buy back the axis already being
# defended at the expense of the ones actually being lost.
#
# So the default uses the scores, not the arguments. This is a real negative
# result about this architecture and it belongs in the write-up, not buried in
# a constant: the specialists' measurable contribution is constraint
# enforcement, not search guidance.
FEEDBACK_RULE = "weakest"


def evaluate(smiles: str, scoring_context=None,
             bbb_goal: str = "penetrate") -> dict:
    """One molecule as the loop passes it around: raw, normalised, provenance."""
    raw = oracles.score(smiles, context=scoring_context)
    normalised = normalize(raw)
    if bbb_goal == "avoid":
        normalised["bbb"] = 1.0 - normalised["bbb"]
    elif bbb_goal == "neutral":
        normalised["bbb"] = 0.5
    return {
        "smiles": canonical(smiles),
        "raw": {axis: raw[axis] for axis in AXES},
        "normalised": normalised,
    }


def pareto_front(points: list[dict], axes: tuple[str, ...] = PARETO_AXES) -> list[dict]:
    """Non-dominated subset on `axes`, all maximised. O(n^2); n is tiny here."""
    front = []
    for point in points:
        dominated = any(
            other is not point
            and all(other["normalised"][a] >= point["normalised"][a] for a in axes)
            and any(other["normalised"][a] > point["normalised"][a] for a in axes)
            for other in points
        )
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda p: p["normalised"][axes[0]])


def hypervolume(points: list[dict], axes: tuple[str, ...] = PARETO_AXES) -> float:
    """Area dominated by the front over the origin. Moves when the front moves."""
    front = pareto_front(points, axes)
    if not front:
        return 0.0
    # 2-D sweep: walk descending in x, accumulating each new band of y.
    area = 0.0
    max_y = 0.0
    for point in sorted(front, key=lambda p: p["normalised"][axes[0]], reverse=True):
        x = point["normalised"][axes[0]]
        y = point["normalised"][axes[1]]
        if y > max_y:
            area += x * (y - max_y)
            max_y = y
    return area


def _axes_under_dispute(reviews: list[dict], parent: dict | None = None,
                        rule: str | None = None) -> tuple[str, ...]:
    """The axes whose agents objected or vetoed, to steer the next proposal.

    An agent that got what it wanted has nothing to redirect. An agent that
    objected is naming the axis the next batch of edits should attack, and a
    veto is the strongest such signal on the board.
    """
    rule = (rule or os.environ.get("FEEDBACK_RULE") or FEEDBACK_RULE).lower()
    if rule == "none":
        return ()

    disputed: list[str] = []
    for review in reviews:
        if review["verdict"] == "support":
            continue
        for axis in _REVIEW_AXES.get(review["agent"], ()):
            if axis not in disputed:
                disputed.append(axis)
    if rule == "disputed":
        return tuple(disputed)

    if parent is None:
        return tuple(disputed)
    weakest = [
        axis for axis, _ in sorted(parent["normalised"].items(), key=lambda kv: kv[1])
    ]
    if rule == "weakest":
        return tuple(weakest[:2])
    # "contested-weak" (default): the axes an agent is fighting for AND the
    # incumbent is actually bad at. An objection about an axis already at 0.9
    # is a preference; one about an axis at 0.2 is a problem.
    intersection = [axis for axis in weakest[:3] if axis in disputed]
    return tuple(intersection or weakest[:2])


_REVIEW_AXES = {
    "Affinity": ("affinity",),
    "ADME": ("solubility", "bbb"),
    "Safety": ("herg",),
}


GENERIC_SYSTEM = (
    "You are a medicinal chemist optimising a DRD2 ligand. Weigh potency, "
    "solubility, brain penetration, hERG risk and synthetic accessibility together "
    "and pick the best overall molecule."
)


def _generic_decision(parent: dict, candidates: list[dict],
                      constrained: bool = False,
                      program: dict | None = None) -> tuple[list[dict], dict]:
    """The ablation arms: one agent weighing everything, with or without the
    hard constraints.

    `constrained=False` is the naive baseline: one agent told to balance five
    properties. `constrained=True` keeps the single agent but bolts the hERG
    veto and the affinity floor on as filters. The gap between the two says how
    much of the specialist architecture's advantage is deliberation and how
    much is just having hard constraints at all -- which is the question a
    reviewer will ask, so it is better to have the number than to be asked for it.
    """
    from core.llm import complete_json

    from core.contract import HERG_VETO_THRESHOLD

    pool = candidates
    vetoed: list[str] = []
    if constrained:
        vetoed = [
            c["smiles"] for c in candidates
            if c["normalised"]["herg"] < HERG_VETO_THRESHOLD
        ]
        pool = [c for c in candidates if c["smiles"] not in vetoed]
        pool = [c for c in pool if c["normalised"]["affinity"] >= AFFINITY_FLOOR]
        if not pool:
            # Hard means hard. An earlier version fell back to the best
            # available here, which made this arm identical to the
            # unconstrained one and the whole comparison meaningless.
            return (
                [{
                    "agent": "Generalist",
                    "verdict": "support",
                    "reason": "No candidate satisfies the hERG and affinity constraints.",
                    "preferred": parent["smiles"],
                }],
                {
                    "chosen": parent["smiles"],
                    "rationale": "Held the incumbent: no candidate met both constraints.",
                    "held_parent": True,
                    "forced_second_best": False,
                    "vetoed": vetoed,
                },
            )

    ranked = sorted(
        pool, key=lambda c: agents.consensus_score(c["normalised"]), reverse=True
    )
    chosen = ranked[0]
    review = {
        "agent": "Generalist",
        "verdict": "support",
        "reason": (
            "Weighed all five properties together and took the best overall at "
            f"{agents.consensus_score(chosen['normalised']):.2f}."
        ),
        "preferred": chosen["smiles"],
    }
    decision = {
        "chosen": chosen["smiles"],
        "rationale": review["reason"],
        "held_parent": False,
        "forced_second_best": False,
        "vetoed": vetoed,
    }
    program = program or {}
    target = program.get("target") or "DRD2"
    system = (
        f"You are a medicinal chemist optimizing a lead against {target}. "
        f"{program.get('rationale', '')} {program.get('activity_strategy', '')} "
        "Weigh target retention, solubility, the stated BBB exposure goal, hERG "
        "risk, and synthetic accessibility."
    )
    payload = complete_json(
        system=system if program else GENERIC_SYSTEM,
        user=(
            f"Parent: {parent['smiles']}\n\nCandidates:\n"
            + agents._candidate_table(candidates)
            + "\n\nPick one and justify it in three sentences.\n"
            'Reply with JSON only: {"chosen": "<candidate SMILES>", "rationale": "..."}'
        ),
    )
    if payload:
        rationale = payload.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            decision["rationale"] = rationale.strip()
            review["reason"] = rationale.strip()
        pick = payload.get("chosen")
        if isinstance(pick, str) and any(c["smiles"] == pick for c in candidates):
            decision["chosen"] = pick
    return [review], decision


def run(seed: str = HALOPERIDOL, rounds: int = 5, mode: str = "specialists",
        n_candidates: int = 6, rng_seed: int | None = 0,
        verbose: bool = True, target: str = "DRD2",
        compound_name: str = "Haloperidol", rationale: str | None = None,
        objective: str | None = None, bbb_goal: str = "penetrate",
        source: str = "Curated example") -> dict:
    """Run the loop and return a record suitable for `demo_run.json`."""
    target = (target or "").strip()
    if not target:
        raise ValueError("a target or phenotype is required")
    if bbb_goal not in {"penetrate", "avoid", "neutral"}:
        raise ValueError("bbb_goal must be 'penetrate', 'avoid', or 'neutral'")
    rationale = (rationale or "").strip() or (
        f"{compound_name} is the starting structure for a program against {target}."
    )
    objective = (objective or "").strip() or (
        f"Preserve target-relevant chemistry for {target} while improving "
        "solubility, hERG safety, and synthetic accessibility."
    )
    scoring_context = oracles.for_program(seed, target)
    program = {
        "compound": compound_name,
        "source": source,
        "target": target,
        "rationale": rationale,
        "objective": objective,
        "bbb_goal": bbb_goal,
        "activity_strategy": scoring_context.activity_note,
    }
    specialists = agents.specialists_for_program(program)

    started = time.time()
    parent = evaluate(seed, scoring_context, bbb_goal)
    history = [dict(parent, round=0, origin="seed")]
    seen = {parent["smiles"]}
    round_records = []
    notes: list[str] = []
    prefer_axes: tuple[str, ...] = ()

    for round_index in range(1, rounds + 1):
        context = {
            "round": round_index,
            "scores": parent["normalised"],
            "agent_notes": notes,
            "n_candidates": n_candidates,
            "seed": None if rng_seed is None else rng_seed + round_index,
            # What the specialists argued about last round steers which
            # transforms are tried this round. Without this the agents are
            # commentary: the orchestrator's pick is already a deterministic
            # function of the scores, so the arguments would change nothing
            # about where the search goes next. This is the one wire that
            # makes the architecture do work rather than narrate it.
            "prefer_axes": prefer_axes or tuple(
                axis
                for axis, value in sorted(parent["normalised"].items(), key=lambda kv: kv[1])[:2]
            ),
            "target": target,
            "target_rationale": rationale,
            "objective": objective,
            "bbb_goal": bbb_goal,
            "activity_strategy": scoring_context.activity_note,
        }
        proposals = edits.propose(parent["smiles"], context)
        mutation_log = dict(edits.MUTATION_LOG)

        candidates = []
        for smiles in proposals:
            try:
                candidate = evaluate(smiles, scoring_context, bbb_goal)
            except ValueError:
                continue  # belt and braces; propose() already validated
            transform, intent = mutation_log.get(smiles, ("unknown", ""))
            candidate["transform"] = transform
            candidate["intent"] = intent
            candidates.append(candidate)
        if not candidates:
            break

        leader = max(candidates, key=lambda c: agents.consensus_score(c["normalised"]))

        if mode.startswith("generic"):
            reviews, decision = _generic_decision(
                parent,
                candidates,
                constrained=(mode == "generic_constrained"),
                program=program,
            )
        else:
            reviews = [
                specialist.review(parent, candidates, leader)
                for specialist in specialists
            ]
            decision = agents.orchestrate(
                parent,
                candidates,
                reviews,
                # The incumbent walks: holding it every round would leave the
                # front to grow out of the archive while the agents' choices
                # changed nothing about where the search went next. The
                # affinity floor and safety's veto are what keep the walk
                # honest, not a refusal to move.
                allow_parent=False,
                accept_tolerance=ACCEPT_TOLERANCE,
                affinity_floor=AFFINITY_FLOOR,
                program=program,
            )

        for candidate in candidates:
            if candidate["smiles"] not in seen:
                seen.add(candidate["smiles"])
                history.append(
                    dict(
                        candidate,
                        round=round_index,
                        origin="vetoed" if candidate["smiles"] in decision["vetoed"]
                        else "candidate",
                    )
                )

        round_records.append(
            {
                "round": round_index,
                "parent": parent,
                "candidates": candidates,
                "leader": leader["smiles"],
                "reviews": reviews,
                "decision": decision,
                "pareto_hypervolume": hypervolume(history),
            }
        )

        if verbose:
            _print_round(round_records[-1])

        notes = [f"{r['agent']} [{r['verdict']}]: {r['reason']}" for r in reviews]
        prefer_axes = _axes_under_dispute(reviews, parent)
        chosen = next(
            (c for c in candidates if c["smiles"] == decision["chosen"]), None
        )
        parent = chosen if chosen is not None else parent

    scored_history = [h for h in history if h["origin"] != "vetoed"]
    best = max(
        scored_history or history,
        key=lambda h: agents.consensus_score(h["normalised"]),
    )
    seed_record = history[0]
    return {
        "target": target,
        "program": program,
        "seed": canonical(seed),
        "mode": mode,
        "rounds": len(round_records),
        "provenance": scoring_context.provenance(),
        "affinity_floor": AFFINITY_FLOOR,
        "affinity_seed": seed_record["normalised"]["affinity"],
        "affinity_final": parent["normalised"]["affinity"],
        "affinity_held": parent["normalised"]["affinity"] >= AFFINITY_FLOOR,
        "weights": agents.ORCHESTRATOR_WEIGHTS,
        "pareto_axes": list(PARETO_AXES),
        "history": history,
        "round_records": round_records,
        "final": parent,
        "best": best,
        "hypervolume_initial": hypervolume([h for h in history if h["round"] == 0]),
        "hypervolume_final": hypervolume(history),
        "wall_seconds": round(time.time() - started, 2),
    }


def _print_round(record: dict) -> None:
    print(f"\n{'=' * 72}\nROUND {record['round']}")
    print(f"  incumbent {record['parent']['smiles']}")
    for candidate in record["candidates"]:
        norm = candidate["normalised"]
        flag = " [VETOED]" if candidate["smiles"] in record["decision"]["vetoed"] else ""
        print(
            f"  - aff {norm['affinity']:.2f} sol {norm['solubility']:.2f} "
            f"bbb {norm['bbb']:.2f} hERG-safe {norm['herg']:.2f} sa {norm['sa']:.2f}  "
            f"{candidate['transform']}{flag}"
        )
    print()
    for review in record["reviews"]:
        marker = {"support": " ", "object": "!", "veto": "X"}.get(review["verdict"], " ")
        print(f"  {marker} {review['agent']:9s} [{review['verdict']:7s}] {review['reason']}")
    print(f"\n  ORCHESTRATOR -> {record['decision']['chosen']}")
    print(f"  {record['decision']['rationale']}")
    print(f"  pareto hypervolume: {record['pareto_hypervolume']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="target-aware multi-agent lead optimisation"
    )
    parser.add_argument("--seed", default=HALOPERIDOL, help="seed SMILES")
    parser.add_argument("--compound-name", default="Haloperidol")
    parser.add_argument("--target", default="DRD2")
    parser.add_argument("--rationale")
    parser.add_argument("--objective")
    parser.add_argument(
        "--bbb-goal",
        choices=("penetrate", "avoid", "neutral"),
        default="penetrate",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("specialists", "generic", "generic_constrained"),
        default="specialists",
    )
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out", default="demo_run.json", help="where to cache the run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    record = run(
        seed=args.seed,
        rounds=args.rounds,
        mode=args.mode,
        n_candidates=args.candidates,
        rng_seed=args.rng_seed,
        verbose=not args.quiet,
        target=args.target,
        compound_name=args.compound_name,
        rationale=args.rationale,
        objective=args.objective,
        bbb_goal=args.bbb_goal,
    )
    Path(args.out).write_text(json.dumps(record, indent=2))

    prov = record["provenance"]
    print(f"\n{'=' * 72}")
    if prov["any_surrogate"]:
        print("WARNING: at least one axis is a SURROGATE, not a trained model.")
        print(f"  {prov}")
    print(
        f"pareto hypervolume ({' x '.join(record['pareto_axes'])}) "
        f"{record['hypervolume_initial']:.3f} -> {record['hypervolume_final']:.3f}"
    )
    verdict = "held" if record["affinity_held"] else "LOST"
    print(
        f"target retention {record['affinity_seed']:.2f} -> "
        f"{record['affinity_final']:.2f} "
        f"(floor {record['affinity_floor']:.2f}: {verdict})"
    )
    if not record["affinity_held"]:
        print("  The front moved by spending the potency that made the seed a lead.")
    print(f"{record['wall_seconds']}s  cached to {args.out}")


if __name__ == "__main__":
    main()
