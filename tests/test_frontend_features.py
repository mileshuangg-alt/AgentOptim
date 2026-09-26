"""Integration checks for the scientist-facing UI helpers."""

from __future__ import annotations

import json
from pathlib import Path

from core.drugs import HALOPERIDOL, IBUPROFEN, search
from frontend.molecule3d import molecule_payload, viewer_html
from frontend.science import candidate_reviews, recommended_assays


def test_drug_lookup_supports_generic_and_brand_names():
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    assert search("haloperidol") == [HALOPERIDOL]
    assert search("Haldol") == [HALOPERIDOL]
    assert search("Advil") == [IBUPROFEN]
    assert IBUPROFEN.optimizable is False
    assert "COX" in IBUPROFEN.limitation
    assert rdMolDescriptors.CalcMolFormula(
        Chem.MolFromSmiles(IBUPROFEN.smiles)
    ) == "C13H18O2"


def test_3d_viewer_has_coordinates_and_precise_change_highlights():
    parent = "c1ccccc1Cl"
    child = "c1ccccc1F"
    payload = molecule_payload(child, parent)
    assert len(payload["atoms"]) == 7
    assert 0 < payload["changed_count"] < len(payload["atoms"])
    assert "canvas" in viewer_html(child, parent)


def test_candidate_review_explains_changes_and_rejections():
    record = json.loads(Path("demo_run.json").read_text())["round_records"][0]
    reviews = candidate_reviews(record)
    assert [review["agent"] for review in reviews] == [
        "Affinity",
        "ADME",
        "Safety",
        "Orchestrator",
    ]
    assert all(
        candidate["change"] and candidate["reason"]
        for review in reviews
        for candidate in review["candidates"]
    )
    assert any(
        candidate["disposition"] == "Vetoed"
        for candidate in reviews[2]["candidates"]
    )


def test_round_assays_always_start_with_identity_and_purity():
    record = json.loads(Path("demo_run.json").read_text())["round_records"][0]
    assays = recommended_assays(record)
    assert assays[0]["name"] == "LC–MS identity and purity"
    assert 1 <= len(assays) <= 4
