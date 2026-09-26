"""Checks for the dependency-free presentation frontend."""

from __future__ import annotations

import json
from pathlib import Path

from frontend.static_adapter import to_static_run


ROOT = Path(__file__).resolve().parents[1]


def test_static_frontend_has_runnable_cached_drugs():
    script = (ROOT / "static" / "app.js").read_text()
    markup = (ROOT / "static" / "index.html").read_text()
    catalogue = json.loads((ROOT / "static_runs" / "drugs.json").read_text())

    assert "static_runs/haloperidol.json" in script
    assert "api/pubchem/compound" in script
    assert 'fetch("../api/optimize"' in script
    assert 'id="programTarget"' in markup
    assert 'id="programRationale"' in markup
    assert 'id="programObjective"' in markup
    assert {drug["name"] for drug in catalogue} == {"Haloperidol", "Ibuprofen"}
    for filename in ("haloperidol.json", "ibuprofen.json"):
        run = json.loads((ROOT / "static_runs" / filename).read_text())
        assert len(run["rounds"]) == 5


def test_static_hero_prevents_narrow_screen_overlap():
    styles = (ROOT / "static" / "style.css").read_text()

    assert ".hero .round-label{position:static" in styles
    assert "grid-template-columns:110px minmax(0,1fr)" in styles
    assert ".hero code{font-size:11px" in styles


def test_native_run_converts_to_static_round_contract():
    native = json.loads((ROOT / "demo_run.json").read_text())
    converted = to_static_run(native)

    assert converted["seed_name"] == "Haloperidol"
    assert len(converted["rounds"]) == 5
    assert all(round_record["structure_3d"] for round_record in converted["rounds"])
    assert all(
        round_record["recommended_assays"]
        and round_record["agent_candidate_review"]
        for round_record in converted["rounds"]
    )
    assert all(
        review["confidence"] is not None
        for round_record in converted["rounds"]
        for review in round_record["agent_candidate_review"]
    )
