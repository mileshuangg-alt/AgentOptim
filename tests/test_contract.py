"""Contract invariants. These are the ones that would silently ruin a demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import agents, edits
from core.contract import AXES, HERG_VETO_THRESHOLD, canonical, is_valid, normalize

HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"


# --- normalize -------------------------------------------------------------

def test_normalize_returns_all_axes_in_unit_range():
    raw = {"affinity": 0.8, "solubility": -4.0, "bbb": 0.7, "herg": 0.3, "sa": 3.0}
    result = normalize(raw)
    assert set(result) == set(AXES)
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_herg_inverts():
    """High predicted blockade must normalise low. Getting this backwards would
    make the safety agent veto the safest molecules, and nothing would crash."""
    safe = normalize({"affinity": 0, "solubility": -4, "bbb": 0, "herg": 0.05, "sa": 3})
    risky = normalize({"affinity": 0, "solubility": -4, "bbb": 0, "herg": 0.95, "sa": 3})
    assert safe["herg"] > risky["herg"]
    assert safe["herg"] == pytest.approx(0.95)


def test_sa_inverts():
    easy = normalize({"affinity": 0, "solubility": -4, "bbb": 0, "herg": 0, "sa": 1.5})
    hard = normalize({"affinity": 0, "solubility": -4, "bbb": 0, "herg": 0, "sa": 8.0})
    assert easy["sa"] > hard["sa"]


def test_solubility_increases_with_logs():
    low = normalize({"affinity": 0, "solubility": -7.0, "bbb": 0, "herg": 0, "sa": 3})
    high = normalize({"affinity": 0, "solubility": -2.0, "bbb": 0, "herg": 0, "sa": 3})
    assert high["solubility"] > low["solubility"]


def test_normalize_clips_out_of_range_inputs():
    result = normalize(
        {"affinity": 1.4, "solubility": 99.0, "bbb": -0.2, "herg": -0.1, "sa": 42.0}
    )
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_normalize_rejects_incomplete_records():
    with pytest.raises(KeyError):
        normalize({"affinity": 0.5})


def test_normalize_ignores_extra_keys():
    """score() adds a provenance key; normalize must not choke on it."""
    raw = {"affinity": 0.5, "solubility": -4, "bbb": 0.5, "herg": 0.5, "sa": 3,
           "provenance": {"affinity": "whatever"}}
    assert set(normalize(raw)) == set(AXES)


# --- validity --------------------------------------------------------------

@pytest.mark.parametrize("smiles", [HALOPERIDOL, "CCO", "c1ccccc1", "CC(=O)Nc1ccccc1"])
def test_is_valid_accepts_real_molecules(smiles):
    assert is_valid(smiles)


@pytest.mark.parametrize(
    "smiles",
    ["", "   ", "not_a_molecule", "c1ccccc", "C(C(C", "CN(C)(C)(C)C", None, 42],
)
def test_is_valid_rejects_junk(smiles):
    assert not is_valid(smiles)


def test_canonical_is_idempotent_and_collapses_equivalent_smiles():
    assert canonical("C1=CC=CC=C1") == canonical("c1ccccc1")
    assert canonical(canonical(HALOPERIDOL)) == canonical(HALOPERIDOL)


def test_canonical_raises_on_invalid():
    with pytest.raises(ValueError):
        canonical("not_a_molecule")


# --- propose ---------------------------------------------------------------

def test_propose_returns_three_to_eight_valid_distinct_children(monkeypatch):
    monkeypatch.setenv("EDIT_ENGINE", "mutation")
    children = edits.propose(HALOPERIDOL, {"round": 1, "scores": {}, "agent_notes": []})
    assert 3 <= len(children) <= 8
    assert len(set(children)) == len(children)
    assert all(is_valid(smiles) for smiles in children)
    assert canonical(HALOPERIDOL) not in children


def test_propose_is_deterministic_for_a_fixed_seed(monkeypatch):
    monkeypatch.setenv("EDIT_ENGINE", "mutation")
    context = {"round": 1, "scores": {}, "agent_notes": [], "seed": 7}
    assert edits.propose(HALOPERIDOL, context) == edits.propose(HALOPERIDOL, context)


def test_propose_rejects_an_invalid_parent(monkeypatch):
    monkeypatch.setenv("EDIT_ENGINE", "mutation")
    with pytest.raises(ValueError):
        edits.propose("not_a_molecule", {"round": 1})


def test_every_transform_is_a_parseable_reaction():
    from rdkit.Chem import AllChem

    for name, smarts, _intent, axis in edits.TRANSFORMS:
        assert AllChem.ReactionFromSmarts(smarts) is not None, name
        assert axis in AXES, name


def test_mutations_stay_within_the_heavy_atom_ceiling(monkeypatch):
    from rdkit import Chem

    monkeypatch.setenv("EDIT_ENGINE", "mutation")
    for smiles in edits.propose(HALOPERIDOL, {"round": 1}):
        assert Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() <= edits._MAX_HEAVY_ATOMS


# --- agents ----------------------------------------------------------------

def _candidate(smiles, **norm):
    base = {axis: 0.5 for axis in AXES}
    base.update(norm)
    return {"smiles": smiles, "normalised": base, "raw": {}}


def test_safety_vetoes_everything_over_the_herg_line(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", herg=0.9)
    candidates = [
        _candidate("CCC", herg=HERG_VETO_THRESHOLD - 0.01),
        _candidate("CCCC", herg=HERG_VETO_THRESHOLD + 0.01),
    ]
    review = agents.SAFETY_AGENT.review(parent, candidates, candidates[0])
    assert review["verdict"] == "veto"
    assert review["vetoed"] == ["CCC"]


def test_orchestrator_cannot_choose_a_vetoed_molecule(monkeypatch):
    """The whole point of the veto. If this regresses, the demo is a lie."""
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", affinity=0.5)
    best_but_toxic = _candidate("CCC", affinity=1.0, bbb=1.0, solubility=1.0, herg=0.01)
    survivor = _candidate("CCCC", affinity=0.7, herg=0.9)
    candidates = [best_but_toxic, survivor]

    reviews = [specialist.review(parent, candidates, best_but_toxic)
               for specialist in agents.SPECIALISTS]
    decision = agents.orchestrate(parent, candidates, reviews)

    assert "CCC" in decision["vetoed"]
    assert decision["chosen"] != "CCC"
    assert decision["chosen"] == "CCCC"
    assert decision["forced_second_best"] is True


def test_orchestrator_holds_the_parent_when_everything_is_vetoed(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", herg=0.9)
    candidates = [_candidate("CCC", herg=0.01), _candidate("CCCC", herg=0.02)]
    reviews = [agents.SAFETY_AGENT.review(parent, candidates, candidates[0])]
    decision = agents.orchestrate(parent, candidates, reviews)
    assert decision["chosen"] == "CCO"
    assert decision["held_parent"] is True


def test_orchestrator_respects_the_affinity_floor(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", affinity=0.9)
    candidates = [
        _candidate("CCC", affinity=0.1, solubility=1.0, bbb=1.0, herg=1.0),
        _candidate("CCCC", affinity=0.8, solubility=0.2, bbb=0.2, herg=0.9),
    ]
    reviews = [specialist.review(parent, candidates, candidates[0])
               for specialist in agents.SPECIALISTS]

    # "CCC" wins on every axis except affinity, where it is far below the floor.
    # It must not be selected under either policy: holding the parent is a valid
    # outcome, selecting a molecule that has stopped binding DRD2 is not.
    held = agents.orchestrate(parent, candidates, reviews, affinity_floor=0.6)
    assert held["chosen"] != "CCC"

    walking = agents.orchestrate(
        parent, candidates, reviews, allow_parent=False, affinity_floor=0.6
    )
    assert walking["chosen"] == "CCCC", "a molecule below the floor was selected"


def test_affinity_floor_is_reported_when_nothing_clears_it(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", affinity=0.9)
    candidates = [_candidate("CCC", affinity=0.1), _candidate("CCCC", affinity=0.2)]
    decision = agents.orchestrate(
        parent, candidates, [], allow_parent=False, affinity_floor=0.6
    )
    assert decision["chosen"] == "CCO"
    assert decision["affinity_floor_breached"] is True


def test_specialists_disagree_on_a_real_trade_off(monkeypatch):
    """Structural disagreement: different argmaxes, no LLM involved."""
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO")
    potent = _candidate("CCC", affinity=0.95, solubility=0.1, bbb=0.3, herg=0.5)
    developable = _candidate("CCCC", affinity=0.4, solubility=0.9, bbb=0.9, herg=0.5)
    candidates = [potent, developable]

    affinity = agents.AFFINITY_AGENT.review(parent, candidates, potent)
    adme = agents.ADME_AGENT.review(parent, candidates, potent)
    assert affinity["preferred"] == "CCC"
    assert adme["preferred"] == "CCCC"
    assert affinity["preferred"] != adme["preferred"]


def test_every_specialist_returns_the_contract_keys(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO")
    candidates = [_candidate("CCC"), _candidate("CCCC", affinity=0.9)]
    for specialist in agents.SPECIALISTS:
        review = specialist.review(parent, candidates, candidates[1])
        assert {
            "verdict", "reason", "preferred", "confidence", "confidence_basis"
        } <= set(review)
        assert review["verdict"] in ("support", "object", "veto")
        assert review["preferred"] in ("CCC", "CCCC")
        assert review["reason"].strip()
        assert 0.5 <= review["confidence"] <= 0.95
        assert review["confidence_basis"].strip()


def test_agent_confidence_increases_with_score_separation():
    assert agents.decision_confidence(0.15) > agents.decision_confidence(0.02)
    assert agents.decision_confidence(1.0) == 0.95


# --- fixtures --------------------------------------------------------------

def test_fixtures_match_the_contract_format():
    fixtures = json.loads(Path("fixtures.json").read_text())
    assert len(fixtures["molecules"]) == 5
    for molecule in fixtures["molecules"]:
        assert is_valid(molecule["smiles"])
        assert set(molecule["raw"]) == set(AXES)
        assert set(molecule["normalised"]) == set(AXES)
        recomputed = normalize(molecule["raw"])
        for axis in AXES:
            assert recomputed[axis] == pytest.approx(
                molecule["normalised"][axis], abs=1e-3
            ), f"{molecule['name']} {axis} normalisation disagrees with the contract"


# --- the relative veto -----------------------------------------------------

def test_absolute_line_is_used_when_the_lead_is_clean(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    clean_parent = _candidate("CCO", herg=0.80)
    line, rule = agents.SAFETY_AGENT.effective_line(clean_parent)
    assert rule == "absolute"
    assert line == HERG_VETO_THRESHOLD


def test_relative_line_is_used_when_the_lead_is_already_liable(monkeypatch):
    """The haloperidol case: the real hERG model scores the seed at 0.865
    blockade, so an absolute line vetoes the seed and every analogue of it."""
    monkeypatch.setenv("AGENT_LLM", "off")
    liable_parent = _candidate("CCO", herg=0.135)
    line, rule = agents.SAFETY_AGENT.effective_line(liable_parent)
    assert rule == "relative"
    assert line == pytest.approx(0.135)


def test_a_liable_lead_does_not_veto_its_own_analogues_wholesale(monkeypatch):
    """Regression test for a demo that killed itself in round 1."""
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", herg=0.135)
    candidates = [
        _candidate("CCC", herg=0.25),    # safer than the lead
        _candidate("CCCC", herg=0.14),   # marginally safer
        _candidate("CCCCC", herg=0.11),  # worse than the lead
    ]
    review = agents.SAFETY_AGENT.review(parent, candidates, candidates[0])
    assert review["vetoed"] == ["CCCCC"], "the veto should discriminate, not blanket"
    assert len(review["vetoed"]) < len(candidates)


def test_relative_veto_still_blocks_deepening_a_known_liability(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "off")
    parent = _candidate("CCO", herg=0.135)
    candidates = [_candidate("CCC", herg=0.05), _candidate("CCCC", herg=0.02)]
    review = agents.SAFETY_AGENT.review(parent, candidates, candidates[0])
    assert review["verdict"] == "veto"
    assert set(review["vetoed"]) == {"CCC", "CCCC"}
