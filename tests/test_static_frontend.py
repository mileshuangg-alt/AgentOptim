"""Checks for the dependency-free presentation frontend."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_frontend_has_runnable_cached_drugs():
    script = (ROOT / "static" / "app.js").read_text()
    catalogue = json.loads((ROOT / "static_runs" / "drugs.json").read_text())

    assert "static_runs/haloperidol.json" in script
    assert {drug["name"] for drug in catalogue} == {"Haloperidol", "Ibuprofen"}
    for filename in ("haloperidol.json", "ibuprofen.json"):
        run = json.loads((ROOT / "static_runs" / filename).read_text())
        assert len(run["rounds"]) == 5


def test_static_hero_prevents_narrow_screen_overlap():
    styles = (ROOT / "static" / "style.css").read_text()

    assert ".hero .round-label{position:static" in styles
    assert "grid-template-columns:110px minmax(0,1fr)" in styles
    assert ".hero code{font-size:11px" in styles
