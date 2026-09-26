"""Four agents: three specialists and an orchestrator.

The disagreement here is structural, not theatrical. Each specialist takes the
argmax of a *different* objective over the same candidate set, and on a real
trade-off those argmaxes differ -- polar groups buy solubility and lose BBB,
lipophilic bulk buys affinity and buys hERG risk with it. So the agents fight
even with the LLM switched off, and the LLM's job is to articulate the conflict
rather than to manufacture it.

Safety is not a voter. It holds a veto: any candidate over the hERG line is
removed from the orchestrator's choices, however much the others wanted it.
That asymmetry is deliberate -- hERG is a programme kill criterion in real
discovery, not a property you trade away for potency.

Each specialist returns verdict, reason, preferred molecule, and bounded
score-separation confidence; safety additionally returns vetoed molecules. The
orchestrator returns its chosen molecule, rationale, and the same confidence
metadata.
"""

from __future__ import annotations

from core.contract import (
    HERG_RELATIVE_MARGIN,
    HERG_VETO_MODE,
    HERG_VETO_THRESHOLD,
)
from core.llm import complete_json

# The orchestrator's weighting over normalised axes. Affinity carries the most
# weight because the seed's potency is the one thing this run must not spend:
# it is the property with no headroom, so it can only be lost. SA is a
# tie-breaker, not a goal.
ORCHESTRATOR_WEIGHTS = {
    "affinity": 0.38,
    "solubility": 0.20,
    "herg": 0.20,
    "bbb": 0.17,
    "sa": 0.05,
}

# How far an axis may slip below the parent before a specialist objects.
OBJECTION_TOLERANCE = 0.02


def consensus_score(normalised: dict) -> float:
    """Weighted mean of the five normalised axes."""
    return sum(weight * normalised[axis] for axis, weight in ORCHESTRATOR_WEIGHTS.items())


def _axis_score(normalised: dict, axes: tuple[str, ...]) -> float:
    return sum(normalised[axis] for axis in axes) / len(axes)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def decision_confidence(margin: float) -> float:
    """Map score separation to a bounded decision-confidence heuristic.

    This describes how decisively an agent's scoring rule separates its choice
    from the alternatives. It is not calibrated model or experimental
    uncertainty, so it deliberately tops out below 1.0.
    """
    scaled = min(max(float(margin), 0.0) / 0.20, 1.0)
    return round(0.50 + 0.45 * scaled, 3)


def ranked_choice_confidence(candidates, scorer, selected: str,
                             score_label: str = "score") -> dict:
    """Confidence metadata for a ranked choice over the candidate set."""
    chosen = next((candidate for candidate in candidates
                   if candidate["smiles"] == selected), None)
    alternatives = [
        candidate for candidate in candidates if candidate["smiles"] != selected
    ]
    if chosen is None:
        return {
            "confidence": 0.50,
            "confidence_basis": "The selected molecule was not in the scored candidate set.",
        }
    if not alternatives:
        return {
            "confidence": 0.95,
            "confidence_basis": "Only one eligible candidate was available.",
        }

    chosen_score = scorer(chosen)
    runner_up = max(scorer(candidate) for candidate in alternatives)
    margin = chosen_score - runner_up
    display_label = (
        score_label if score_label.isupper() else score_label.title()
    )
    if margin >= 0:
        basis = f"{display_label} leads the next candidate by {_fmt(margin)}."
    else:
        basis = (
            f"The selected candidate trails the {display_label} leader by "
            f"{_fmt(abs(margin))}; confidence is therefore limited."
        )
    return {
        "confidence": decision_confidence(margin),
        "confidence_basis": basis,
    }


class Specialist:
    """A single-objective advocate over one or two normalised axes."""

    def __init__(self, name: str, axes: tuple[str, ...], system: str, label: str):
        self.name = name
        self.axes = axes
        self.system = system
        self.label = label

    def _deterministic(self, parent, candidates, leader) -> dict:
        parent_value = _axis_score(parent["normalised"], self.axes)
        best = max(candidates, key=lambda c: _axis_score(c["normalised"], self.axes))
        best_value = _axis_score(best["normalised"], self.axes)
        leader_value = _axis_score(leader["normalised"], self.axes)

        axes_label = " + ".join(self.axes)
        if leader_value < parent_value - OBJECTION_TOLERANCE:
            verdict = "object"
            reason = (
                f"The field leader gives up {axes_label}: {_fmt(leader_value)} against the "
                f"parent's {_fmt(parent_value)}. My pick holds {_fmt(best_value)}. "
                f"I am not trading {axes_label} backwards this round."
            )
        elif best_value > leader_value + OBJECTION_TOLERANCE:
            verdict = "object"
            reason = (
                f"The leader is acceptable on {axes_label} ({_fmt(leader_value)}) but it is "
                f"not the best available: my pick reaches {_fmt(best_value)}. "
                f"I want the stronger option on the table."
            )
        else:
            verdict = "support"
            reason = (
                f"The leader is also my pick on {axes_label} at {_fmt(best_value)}, "
                f"up from the parent's {_fmt(parent_value)}. No objection."
            )
        return {"verdict": verdict, "reason": reason, "preferred": best["smiles"]}

    def review(self, parent, candidates, leader) -> dict:
        """Deterministic verdict, with the LLM's wording layered on if available."""
        result = self._deterministic(parent, candidates, leader)
        payload = complete_json(
            system=self.system,
            user=_specialist_prompt(self, parent, candidates, leader),
        )
        if payload:
            reason = payload.get("reason")
            if isinstance(reason, str) and reason.strip():
                result["reason"] = reason.strip()
            verdict = payload.get("verdict")
            # The LLM may sharpen support into an objection but cannot award
            # itself a veto -- only the safety agent holds one.
            if verdict in ("support", "object"):
                result["verdict"] = verdict
            preferred = payload.get("preferred")
            if isinstance(preferred, str) and any(
                c["smiles"] == preferred for c in candidates
            ):
                result["preferred"] = preferred
        result.update(
            ranked_choice_confidence(
                candidates,
                lambda candidate: _axis_score(candidate["normalised"], self.axes),
                result["preferred"],
                self.label,
            )
        )
        result["agent"] = self.label
        return result


def _candidate_table(candidates) -> str:
    lines = []
    for index, candidate in enumerate(candidates, 1):
        norm = candidate["normalised"]
        lines.append(
            f"{index}. {candidate['smiles']}\n"
            f"   affinity {_fmt(norm['affinity'])}  solubility {_fmt(norm['solubility'])}  "
            f"bbb {_fmt(norm['bbb'])}  herg-safety {_fmt(norm['herg'])}  "
            f"sa {_fmt(norm['sa'])}"
        )
    return "\n".join(lines)


def _specialist_prompt(agent: Specialist, parent, candidates, leader) -> str:
    parent_norm = parent["normalised"]
    return (
        f"Parent: {parent['smiles']}\n"
        f"   affinity {_fmt(parent_norm['affinity'])}  solubility {_fmt(parent_norm['solubility'])}  "
        f"bbb {_fmt(parent_norm['bbb'])}  herg-safety {_fmt(parent_norm['herg'])}  "
        f"sa {_fmt(parent_norm['sa'])}\n\n"
        f"Candidates this round:\n{_candidate_table(candidates)}\n\n"
        f"The unweighted field leader is: {leader['smiles']}\n\n"
        "All values are normalised 0-1 where higher is better, including "
        "herg-safety (1.0 = no predicted blockade).\n"
        "Argue your axis only. Be specific about the numbers and concise "
        "(two sentences).\n"
        'Reply with JSON only: {"verdict": "support"|"object", "reason": "...", '
        '"preferred": "<one candidate SMILES exactly as written above>"}'
    )


AFFINITY_AGENT = Specialist(
    name="affinity",
    axes=("affinity",),
    label="Affinity",
    system=(
        "You are the affinity specialist on a DRD2 lead optimisation team. "
        "Potency is your only concern. You argue for the strongest binder and you "
        "push back when the team drifts toward weaker, tidier molecules. "
        "You accept that others will constrain you, but you never concede your axis "
        "unprompted."
    ),
)

ADME_AGENT = Specialist(
    name="adme",
    axes=("solubility", "bbb"),
    label="ADME",
    system=(
        "You are the ADME specialist on a DRD2 lead optimisation team. You care about "
        "aqueous solubility and blood-brain-barrier penetration, and you know they "
        "fight each other: polarity buys solubility and loses CNS exposure. A potent "
        "molecule that cannot be dosed or cannot reach the brain is not a lead. "
        "Say plainly when a proposal is undevelopable."
    ),
)


class SafetyAgent:
    """Holds the hERG veto. Removes candidates; does not negotiate."""

    name = "safety"
    label = "Safety"
    axes = ("herg",)
    system = (
        "You are the safety specialist on a DRD2 lead optimisation team, responsible "
        "for hERG cardiotoxicity. hERG blockade is a programme kill criterion, not a "
        "trade-off: QT prolongation has ended real clinical programmes. You do not "
        "bargain potency against it. State your veto in one or two flat, unhedged "
        "sentences."
    )

    def __init__(self, threshold: float = HERG_VETO_THRESHOLD,
                 mode: str = HERG_VETO_MODE,
                 relative_margin: float = HERG_RELATIVE_MARGIN,
                 system: str | None = None):
        self.threshold = threshold
        self.mode = mode
        self.relative_margin = relative_margin
        if system:
            self.system = system

    def effective_line(self, parent) -> tuple[float, str]:
        """The safety floor for this round, and which rule produced it.

        See HERG_VETO_MODE in core.contract for why this is not simply a
        constant: an absolute line vetoes the entire haloperidol series,
        seed included, and leaves the loop with nothing to choose.
        """
        parent_safety = parent["normalised"]["herg"]
        if self.mode == "absolute":
            return self.threshold, "absolute"
        if self.mode == "relative":
            return parent_safety - self.relative_margin, "relative"
        if parent_safety >= self.threshold:
            return self.threshold, "absolute"
        return parent_safety - self.relative_margin, "relative"

    def review(self, parent, candidates, leader) -> dict:
        # The hard rule fires first and always. The LLM only supplies wording.
        line, rule = self.effective_line(parent)
        vetoed = [c["smiles"] for c in candidates if c["normalised"]["herg"] < line]
        survivors = [c for c in candidates if c["smiles"] not in vetoed]
        pool = survivors or candidates
        best = max(pool, key=lambda c: c["normalised"]["herg"])

        blocked_prob = 1.0 - line
        if vetoed:
            verdict = "veto"
            leader_note = (
                " That includes the molecule this round was converging on."
                if leader["smiles"] in vetoed
                else ""
            )
            basis = (
                f"above {_fmt(blocked_prob)}"
                if rule == "absolute"
                else (
                    f"worse than the lead we already have, which is itself at "
                    f"{_fmt(1.0 - parent['normalised']['herg'])} blockade"
                )
            )
            reason = (
                f"Vetoing {len(vetoed)} candidate(s) on predicted hERG blockade "
                f"{basis}.{leader_note} This series already carries a QT liability; "
                f"I will not deepen it for potency. The safest remaining option is "
                f"herg-safety {_fmt(best['normalised']['herg'])}."
            )
        elif best["normalised"]["herg"] < parent["normalised"]["herg"] - OBJECTION_TOLERANCE:
            verdict = "object"
            reason = (
                f"Nothing crosses my line, but the whole batch drifts the wrong way: best "
                f"herg-safety {_fmt(best['normalised']['herg'])} against the parent's "
                f"{_fmt(parent['normalised']['herg'])}. Stop adding lipophilic bulk to a "
                f"basic amine."
            )
        else:
            verdict = "support"
            reason = (
                f"No candidate crosses the hERG line. Best is "
                f"{_fmt(best['normalised']['herg'])} against the parent's "
                f"{_fmt(parent['normalised']['herg'])}."
            )

        payload = complete_json(
            system=self.system,
            user=(
                _specialist_prompt(self, parent, candidates, leader)
                + f"\n\nYou have already ruled that these candidates exceed your hERG "
                f"limit and are removed: {vetoed or 'none'}. Explain that ruling; do not "
                f"revisit it.\n"
                'Use {"verdict": "veto"|"object"|"support", "reason": "...", '
                '"preferred": "..."}'
            ),
        )
        if payload:
            reason_text = payload.get("reason")
            if isinstance(reason_text, str) and reason_text.strip():
                reason = reason_text.strip()

        closest_to_line = min(
            abs(candidate["normalised"]["herg"] - line)
            for candidate in candidates
        )
        return {
            "agent": self.label,
            "verdict": verdict,
            "reason": reason,
            "preferred": best["smiles"],
            "vetoed": vetoed,
            "confidence": decision_confidence(closest_to_line),
                "confidence_basis": (
                    "The closest candidate is "
                    f"{closest_to_line:.3f} normalized hERG units from the binding "
                    f"{rule} veto line."
                ),
        }


SAFETY_AGENT = SafetyAgent()
SPECIALISTS = (AFFINITY_AGENT, ADME_AGENT, SAFETY_AGENT)


def specialists_for_program(program: dict) -> tuple:
    """Create target-aware prompts while retaining deterministic agent logic."""
    target = program.get("target") or "the selected target"
    rationale = program.get("rationale") or "No target rationale was supplied."
    activity_note = program.get("activity_strategy") or ""
    bbb_goal = program.get("bbb_goal", "penetrate")
    exposure = {
        "penetrate": (
            "The program requires CNS exposure, so higher predicted BBB "
            "penetration is desirable."
        ),
        "avoid": (
            "The program is peripheral, so lower predicted BBB penetration is "
            "desirable. The supplied BBB score is already inverted to goal fit."
        ),
        "neutral": (
            "BBB exposure is not a selection objective for this program; its "
            "goal-fit score is held neutral."
        ),
    }.get(bbb_goal, "")
    context = f"Target: {target}. Program rationale: {rationale}"

    affinity = Specialist(
        name="affinity",
        axes=("affinity",),
        label="Affinity",
        system=(
            f"You are the target-activity specialist for a lead program. {context} "
            f"{activity_note} Defend retention of target-relevant chemistry and "
            "state clearly when the activity score is only a structural proxy."
        ),
    )
    adme = Specialist(
        name="adme",
        axes=("solubility", "bbb"),
        label="ADME",
        system=(
            f"You are the ADME specialist for a lead program. {context} "
            f"{exposure} Balance that exposure goal with aqueous solubility."
        ),
    )
    safety = SafetyAgent(
        system=(
            f"You are the safety specialist for a lead program against {target}. "
            "You are responsible for predicted hERG cardiotoxicity. Apply the "
            "program's binding veto exactly and explain it concisely."
        )
    )
    return affinity, adme, safety

ORCHESTRATOR_SYSTEM = (
    "You are the orchestrator of a DRD2 lead optimisation team. Three specialists "
    "have reported: affinity, ADME, and safety. Safety holds a veto and its vetoes "
    "are already applied -- vetoed molecules are gone and you may not choose one. "
    "Your job is to pick one molecule from what remains and explain the trade-off "
    "you accepted in doing so. Name what you gave up. If you are taking a weaker "
    "binder because the stronger one was vetoed, say that in plain words."
)


def orchestrate(parent, candidates, reviews, allow_parent: bool = True,
                accept_tolerance: float = 0.0, affinity_floor: float = 0.0,
                program: dict | None = None) -> dict:
    """Pick one molecule from the non-vetoed candidates and explain the choice.

    Two hard constraints, and they are different in kind:

    - safety's **veto**, which an agent casts on the molecules it judges;
    - the **affinity floor**, a programme criterion set in advance by the team,
      applied here rather than argued by the affinity agent.

    Keeping them separate matters. The floor is what stops a multi-objective
    walk from quietly spending the potency that made the seed a lead, and it is
    a number a reviewer can check. Safety's veto is a judgement call an agent
    makes and has to defend in the transcript.
    """
    program = program or {}
    target = program.get("target") or "the selected target"
    activity_label = (
        "target-retention score"
        if "similarity" in (program.get("activity_strategy") or "").lower()
        else "target activity"
    )
    vetoed = set()
    for review in reviews:
        vetoed.update(review.get("vetoed") or [])

    survivors = [c for c in candidates if c["smiles"] not in vetoed]
    floor_breached = False
    if survivors and affinity_floor > 0.0:
        above_floor = [
            c for c in survivors if c["normalised"]["affinity"] >= affinity_floor
        ]
        if above_floor:
            survivors = above_floor
        else:
            # Nothing clears the potency criterion. Report it rather than pick
            # the least-bad molecule as though the constraint were advisory.
            floor_breached = True
            survivors = []

    if not survivors and floor_breached:
        return {
            "chosen": parent["smiles"],
            "rationale": (
                f"No surviving candidate holds affinity at the programme floor of "
                f"{_fmt(affinity_floor)}. Holding the incumbent: a more soluble, safer "
                f"molecule that loses its {activity_label} for {target} is not a lead, "
                f"and this round produced nothing else."
            ),
            "held_parent": True,
            "affinity_floor_breached": True,
            "vetoed": sorted(vetoed),
            "confidence": 0.95,
            "confidence_basis": "A binding affinity-floor rule determined the outcome.",
        }

    if not survivors:
        # Everything was vetoed. Holding the parent is the correct answer: a
        # round that produces no safe molecule has produced no molecule.
        return {
            "chosen": parent["smiles"],
            "rationale": (
                "Every candidate this round was vetoed on hERG. I am holding the "
                "incumbent rather than advancing a cardiotoxicity risk, and the edit "
                "engine should avoid lipophilic additions to the basic amine next round."
            ),
            "held_parent": True,
            "vetoed": sorted(vetoed),
            "confidence": 0.95,
            "confidence_basis": "A binding safety veto removed every candidate.",
        }

    ranked = sorted(survivors, key=lambda c: consensus_score(c["normalised"]), reverse=True)
    chosen = ranked[0]

    parent_value = consensus_score(parent["normalised"])
    if allow_parent and parent_value - accept_tolerance > consensus_score(
        chosen["normalised"]
    ):
        return {
            "chosen": parent["smiles"],
            "rationale": (
                f"No surviving candidate comes within {_fmt(accept_tolerance)} of the "
                f"incumbent on the weighted objective ({_fmt(parent_value)} against "
                f"{_fmt(consensus_score(chosen['normalised']))}). Holding position and "
                f"re-proposing from the same parent."
            ),
            "held_parent": True,
            "vetoed": sorted(vetoed),
            "confidence": 0.95,
            "confidence_basis": "The incumbent-retention rule determined the outcome.",
        }

    # Was the choice forced by the veto? That is the line worth reading aloud.
    best_overall = max(candidates, key=lambda c: consensus_score(c["normalised"]))
    forced = best_overall["smiles"] in vetoed

    objections = [r for r in reviews if r["verdict"] == "object"]
    rationale_parts = [
        f"Chose {chosen['smiles']} at weighted "
        f"{_fmt(consensus_score(chosen['normalised']))}."
    ]
    if forced:
        rationale_parts.append(
            f"This was second best. The leading candidate scored "
            f"{_fmt(consensus_score(best_overall['normalised']))} and safety vetoed it on "
            f"hERG, so it is off the table regardless of its potency."
        )
    for review in objections:
        rationale_parts.append(f"{review['agent']} objected: {review['reason']}")

    result = {
        "chosen": chosen["smiles"],
        "rationale": " ".join(rationale_parts),
        "held_parent": False,
        "forced_second_best": forced,
        "vetoed": sorted(vetoed),
    }

    orchestrator_system = (
        f"You orchestrate a lead optimization program against {target}. "
        f"{program.get('rationale', '')} Three specialists reported on target "
        "retention, ADME, and safety. Safety vetoes are binding. Pick one surviving "
        "candidate and explain the target-relevant trade-off, including the limits "
        f"of the activity evidence. {program.get('activity_strategy', '')}"
    )
    payload = complete_json(
        system=orchestrator_system if program else ORCHESTRATOR_SYSTEM,
        user=(
            f"Parent: {parent['smiles']}\n\n"
            f"Surviving candidates:\n{_candidate_table(survivors)}\n\n"
            f"Vetoed and unavailable: {sorted(vetoed) or 'none'}\n\n"
            "Specialist reports:\n"
            + "\n".join(
                f"- {r['agent']} [{r['verdict']}] wants {r['preferred']}: {r['reason']}"
                for r in reviews
            )
            + f"\n\nThe weighted objective ranks {chosen['smiles']} first among "
            f"survivors. Explain the trade-off you are accepting in three sentences.\n"
            'Reply with JSON only: {"chosen": "<surviving candidate SMILES>", '
            '"rationale": "..."}'
        ),
    )
    if payload:
        rationale = payload.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            result["rationale"] = rationale.strip()
        pick = payload.get("chosen")
        if isinstance(pick, str) and any(c["smiles"] == pick for c in survivors):
            result["chosen"] = pick
    result.update(
        ranked_choice_confidence(
            survivors,
            lambda candidate: consensus_score(candidate["normalised"]),
            result["chosen"],
            "weighted utility",
        )
    )
    return result
