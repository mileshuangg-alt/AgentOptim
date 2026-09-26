from core import admet
from scripts.verify_models import (
    check_featurization,
    check_manifest,
    check_model_schema,
    check_rdkit_sa,
)


def test_committed_admet_artifacts_and_schema():
    ok, message = check_featurization()
    assert ok, message

    ok, details = check_manifest(
        admet.MODELS_DIR, admet.MODELS_DIR / "manifest.json"
    )
    assert ok, details

    ok, details = check_model_schema(admet.XGBoostBackend())
    assert ok, details


def test_rdkit_sa_fragment_model_is_available():
    ok, message = check_rdkit_sa()
    assert ok, message
