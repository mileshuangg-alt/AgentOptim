from core import pubchem
from core.drugs import from_pubchem


def test_pubchem_name_search_encodes_query_and_limits_results(monkeypatch):
    seen = {}

    def fake_fetch(url):
        seen["url"] = url
        return {
            "dictionary_terms": {
                "compound": ["Aspirin", "Aspirin calcium", "Aspirin aluminum"]
            }
        }

    monkeypatch.setattr(pubchem, "_fetch_json", fake_fetch)
    assert pubchem.search_names("acetyl salicylic", limit=2) == [
        "Aspirin",
        "Aspirin calcium",
    ]
    assert "acetyl%20salicylic" in seen["url"]
    assert seen["url"].endswith("limit=2")


def test_pubchem_compound_becomes_an_editable_target_program(monkeypatch):
    monkeypatch.setattr(
        pubchem,
        "_fetch_json",
        lambda _url: {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 2244,
                        "Title": "Aspirin",
                        "MolecularFormula": "C9H8O4",
                        "IUPACName": "2-acetyloxybenzoic acid",
                        "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    }
                ]
            }
        },
    )
    compound = pubchem.fetch_compound("aspirin")
    drug = from_pubchem(compound)

    assert compound.cid == 2244
    assert compound.name == "Aspirin"
    assert drug.pubchem_cid == 2244
    assert drug.optimizable
    assert drug.target == ""
    assert "similarity" in drug.limitation


def test_pubchem_missing_structure_is_reported(monkeypatch):
    monkeypatch.setattr(
        pubchem,
        "_fetch_json",
        lambda _url: {"PropertyTable": {"Properties": [{"CID": 1}]}},
    )
    try:
        pubchem.fetch_compound("unknown")
    except pubchem.PubChemError as exc:
        assert "without a SMILES" in str(exc)
    else:
        raise AssertionError("missing structure should raise PubChemError")


def test_pubchem_salt_uses_largest_fragment_for_optimization(monkeypatch):
    monkeypatch.setattr(
        pubchem,
        "_fetch_json",
        lambda _url: {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 14219,
                        "Title": "Metformin hydrochloride",
                        "SMILES": "CN(C)C(=N)N=C(N)N.Cl",
                    }
                ]
            }
        },
    )
    compound = pubchem.fetch_compound("metformin hydrochloride")
    assert "." not in compound.smiles
    assert "." in compound.original_smiles
