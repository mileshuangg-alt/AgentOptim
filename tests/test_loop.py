"""End-to-end loop behaviour, and the claims the demo makes about it."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from core import loop
from core.contract import AXES, is_valid


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Every test here runs with no network and no API key."""
    monkeypatch.setenv("AGENT_LLM", "off")
    monkeypatch.setenv("EDIT_ENGINE", "mutation")
    monkeypatch.setenv("ADMET_BACKEND", "surrogate")
    monkeypatch.setenv("AFFINITY_ORACLE", "similarity")


@pytest.fixture(scope="module")
def run_record():
    return loop.run(rounds=4, verbose=False, rng_seed=0)


def test_score_returns_five_floats():
    from core.contract import score

    record = score("CCO")
    assert set(AXES) <= set(record)
    assert all(isinstance(record[axis], float) for axis in AXES)


def test_score_rejects_invalid_smiles():
    from core.contract import score

    with pytest.raises(ValueError):
        score("not_a_molecule")


def test_loop_completes_every_round(run_record):
    assert run_record["rounds"] == 4
    assert len(run_record["round_records"]) == 4


def test_every_molecule_the_loop_touched_is_valid(run_record):
    for entry in run_record["history"]:
        assert is_valid(entry["smiles"])


def test_the_safety_veto_fires_and_is_honoured(run_record):
    """The demo's central claim. If no veto ever fires, there is no demo."""
    vetoed_anywhere = [
        smiles
        for record in run_record["round_records"]
        for smiles in record["decision"]["vetoed"]
    ]
    assert vetoed_anywhere, "no candidate was ever vetoed across four rounds"
    for record in run_record["round_records"]:
        assert record["decision"]["chosen"] not in record["decision"]["vetoed"]


def test_agents_actually_disagree(run_record):
    """If all three always support, this is one model with extra steps."""
    verdicts = {
        review["verdict"]
        for record in run_record["round_records"]
        for review in record["reviews"]
    }
    assert verdicts - {"support"}, "no agent ever objected or vetoed"


def test_the_pareto_front_moves(run_record):
    assert run_record["hypervolume_final"] > run_record["hypervolume_initial"]


def test_affinity_is_held_above_the_floor(run_record):
    """A front that moved by spending all the potency has not optimised anything."""
    assert run_record["affinity_held"], (
        f"affinity fell to {run_record['affinity_final']:.2f}, "
        f"below the floor of {run_record['affinity_floor']:.2f}"
    )


def test_run_is_reproducible_for_a_fixed_seed():
    first = loop.run(rounds=2, verbose=False, rng_seed=11)
    second = loop.run(rounds=2, verbose=False, rng_seed=11)
    assert first["final"]["smiles"] == second["final"]["smiles"]


def test_run_record_is_json_serialisable(run_record):
    """The frontend replays this file; an unserialisable field breaks the cache."""
    restored = json.loads(json.dumps(run_record))
    assert restored["final"]["smiles"] == run_record["final"]["smiles"]


def test_arbitrary_target_uses_seed_similarity_and_records_justification():
    ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    record = loop.run(
        seed=ibuprofen,
        rounds=1,
        verbose=False,
        target="COX-2",
        compound_name="Ibuprofen",
        rationale="COX-2 drives inflammatory prostaglandin synthesis.",
        objective="Retain COX-2 activity while improving developability.",
        bbb_goal="avoid",
    )

    assert record["target"] == "COX-2"
    assert record["program"]["rationale"].startswith("COX-2")
    assert record["program"]["bbb_goal"] == "avoid"
    assert record["provenance"]["affinity"] == "seed-similarity"
    assert record["history"][0]["normalised"]["affinity"] == pytest.approx(1.0)
    assert record["history"][0]["normalised"]["bbb"] == pytest.approx(
        1.0 - record["history"][0]["raw"]["bbb"]
    )


def test_provenance_marks_the_surrogate_backend(run_record):
    """The surrogate must never be able to pass itself off as a trained model."""
    assert run_record["provenance"]["any_surrogate"] is True


# --- pareto helpers --------------------------------------------------------

def _point(x, y):
    return {"normalised": {"solubility": x, "herg": y}}


def test_pareto_front_drops_dominated_points():
    points = [_point(0.5, 0.5), _point(0.4, 0.4), _point(0.9, 0.1)]
    front = loop.pareto_front(points, ("solubility", "herg"))
    assert _point(0.4, 0.4)["normalised"] not in [p["normalised"] for p in front]
    assert len(front) == 2


def test_hypervolume_grows_when_a_point_dominates():
    small = loop.hypervolume([_point(0.3, 0.3)], ("solubility", "herg"))
    large = loop.hypervolume([_point(0.3, 0.3), _point(0.6, 0.6)], ("solubility", "herg"))
    assert large > small
    assert small == pytest.approx(0.09)


def test_hypervolume_of_an_empty_archive_is_zero():
    assert loop.hypervolume([], ("solubility", "herg")) == 0.0


# --- ablation --------------------------------------------------------------

def test_generic_mode_runs_without_a_veto():
    """The ablation arm: one agent, no specialists, nothing removed."""
    record = loop.run(rounds=3, mode="generic", verbose=False, rng_seed=0)
    assert record["mode"] == "generic"
    assert all(len(r["reviews"]) == 1 for r in record["round_records"])
    assert not any(r["decision"]["vetoed"] for r in record["round_records"])


def test_cli_writes_a_cached_run(tmp_path):
    out = tmp_path / "run.json"
    result = subprocess.run(
        [sys.executable, "-m", "core.loop", "--rounds", "2", "--quiet",
         "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["rounds"] == 2
