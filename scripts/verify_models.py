"""Reproducibility gate for the inherited ADMET models.

The saved XGBoost estimators require a specific feature shape and column
order. A vector with the right width but a different descriptor order still
returns confident predictions, so merely loading and running the models is not
enough.

This gate checks:

1. featurization shape and determinism;
2. SHA-256 hashes of the committed model artifacts;
3. model feature names and prediction fixtures;
4. optionally, exact feature-table equality with the source repository.

Run the committed checks:

    python -m scripts.verify_models

Compare directly with a local source checkout as well:

    python -m scripts.verify_models --source-repo /path/to/admet-property-prediction
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from core import admet

DEFAULT_TOLERANCE = 1e-6
SOURCE_CASES = (
    "CCO",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "CC(=O)[O-].[Na+]",
)


def check_featurization() -> tuple[bool, str]:
    """Check feature-table shape and determinism before loading a model."""
    try:
        features = admet.featurize("CCO")
    except Exception as exc:
        return False, f"featurize() raised {type(exc).__name__}: {exc}"

    expected_width = admet.MORGAN_BITS + len(admet.DESCRIPTOR_ORDER)
    width = features.shape[1]
    if width != expected_width:
        return False, (
            f"feature width {width}, expected {expected_width} "
            f"({admet.MORGAN_BITS} fingerprint bits + "
            f"{len(admet.DESCRIPTOR_ORDER)} descriptors)"
        )
    if list(features.columns) != admet.FEATURE_NAMES:
        return False, "feature column names or order do not match FEATURE_NAMES"
    if not admet.featurize("CCO").equals(features):
        return False, "featurize() is not deterministic for the same input"
    return True, (
        f"feature vector {width} wide: Morgan r={admet.MORGAN_RADIUS} "
        f"x {admet.MORGAN_BITS} bits + {list(admet.DESCRIPTOR_ORDER)}"
    )


def check_manifest(models_dir: Path, manifest_path: Path) -> tuple[bool, list[str]]:
    """Verify the binaries are the exact source artifacts."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return False, [f"{manifest_path}: {type(exc).__name__}: {exc}"]

    failures: list[str] = []
    details: list[str] = []
    model_records = manifest.get("models", {})
    if not model_records:
        return False, [f"{manifest_path}: no model records"]

    for filename, metadata in model_records.items():
        path = models_dir / filename
        if not path.exists():
            failures.append(f"{filename}: missing")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        want = metadata.get("sha256")
        if got != want:
            failures.append(f"{filename}: SHA-256 {got}, expected {want}")
        else:
            details.append(f"{filename}={got[:12]}…")
    return (False, failures) if failures else (True, details)


def check_model_schema(backend: admet.XGBoostBackend) -> tuple[bool, list[str]]:
    """Require the feature count and names in each estimator to match."""
    failures: list[str] = []
    details: list[str] = []
    for task, model in backend._models.items():
        width = getattr(model, "n_features_in_", None)
        names = list(getattr(model, "feature_names_in_", []))
        if width != len(admet.FEATURE_NAMES):
            failures.append(
                f"{task}: n_features_in_={width}, expected {len(admet.FEATURE_NAMES)}"
            )
        elif names != admet.FEATURE_NAMES:
            failures.append(f"{task}: feature_names_in_ does not match local schema")
        else:
            details.append(f"{task}=2054 named features")
    return (False, failures) if failures else (True, details)


def _source_featurizer(source_repo: Path):
    """Load only source app.py's featurization functions, without running its UI."""
    source_file = source_repo / "app.py"
    tree = ast.parse(source_file.read_text(), filename=str(source_file))
    wanted = {"largest_fragment", "compute_morgan_fp", "featurize_one"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise ValueError(f"{source_file} does not define {sorted(wanted)}")

    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    namespace = {
        "np": np,
        "pd": pd,
        "Chem": Chem,
        "AllChem": AllChem,
        "Descriptors": Descriptors,
        "ConvertToNumpyArray": ConvertToNumpyArray,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(source_file), "exec"), namespace)
    return namespace["largest_fragment"], namespace["featurize_one"]


def check_source_featurization(source_repo: Path) -> tuple[bool, list[str]]:
    """Compare local features with source app.py on drugs and a salt."""
    from rdkit import Chem

    try:
        largest_fragment, source_featurize = _source_featurizer(source_repo)
    except Exception as exc:
        return False, [f"{type(exc).__name__}: {exc}"]

    failures: list[str] = []
    details: list[str] = []
    for smiles in SOURCE_CASES:
        mol = largest_fragment(Chem.MolFromSmiles(smiles))
        source_features, _ = source_featurize(mol)
        local_features = admet.featurize(smiles)
        if not local_features.equals(source_features):
            failures.append(f"{smiles}: local and source DataFrames differ")
        else:
            details.append(f"{smiles}: exact")
    return (False, failures) if failures else (True, details)


def check_rdkit_sa() -> tuple[bool, str]:
    """Check RDKit's bundled SA fragment model and scorer are available."""
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig

        from core.oracles import _sa_scorer

        fragment_model = (
            Path(RDConfig.RDContribDir) / "SA_Score" / "fpscores.pkl.gz"
        )
        if not fragment_model.exists():
            return False, f"missing bundled fragment model: {fragment_model}"
        mol = Chem.MolFromSmiles(SOURCE_CASES[2])
        value = float(_sa_scorer()(mol))
        if not 1.0 <= value <= 10.0:
            return False, f"SA score {value} is outside the documented 1-10 range"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"fpscores.pkl.gz loaded; haloperidol SA={value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the inherited ADMET models")
    parser.add_argument(
        "--expected",
        type=Path,
        default=admet.MODELS_DIR / "expected_predictions.json",
        help="prediction fixture JSON (default: models/expected_predictions.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=admet.MODELS_DIR / "manifest.json",
        help="model artifact manifest (default: models/manifest.json)",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="optional local checkout of tuhinc5203/admet-property-prediction",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    print("=" * 72)
    ok, message = check_featurization()
    print(f"[{'PASS' if ok else 'FAIL'}] featurization: {message}")
    if not ok:
        return 1

    ok, details = check_manifest(admet.MODELS_DIR, args.manifest)
    print(f"[{'PASS' if ok else 'FAIL'}] model artifacts:")
    for detail in details:
        print(f"       {detail}")
    if not ok:
        return 1

    if args.source_repo:
        ok, details = check_source_featurization(args.source_repo)
        print(f"[{'PASS' if ok else 'FAIL'}] source featurization:")
        for detail in details:
            print(f"       {detail}")
        if not ok:
            return 1

    ok, message = check_rdkit_sa()
    print(f"[{'PASS' if ok else 'FAIL'}] RDKit SA model: {message}")
    if not ok:
        return 1

    try:
        backend = admet.XGBoostBackend()
    except Exception as exc:
        print(f"[FAIL] model loading: {type(exc).__name__}: {exc}")
        return 1
    print(f"[PASS] model loading: three models from {backend.models_dir}/")

    ok, details = check_model_schema(backend)
    print(f"[{'PASS' if ok else 'FAIL'}] model schema:")
    for detail in details:
        print(f"       {detail}")
    if not ok:
        return 1

    try:
        expected = json.loads(args.expected.read_text())
    except Exception as exc:
        print(f"[FAIL] prediction fixtures: {type(exc).__name__}: {exc}")
        return 1

    failures = 0
    checked = 0
    for task, cases in expected.items():
        method = getattr(backend, task, None)
        if method is None:
            print(f"[FAIL] unknown task {task!r}")
            failures += 1
            continue
        for smiles, want in cases.items():
            checked += 1
            try:
                got = method(smiles)
            except Exception as exc:
                print(f"[FAIL] {task} {smiles}: {type(exc).__name__}: {exc}")
                failures += 1
                continue
            delta = abs(got - float(want))
            if delta <= args.tolerance:
                print(
                    f"[PASS] {task:11s} {smiles:40s} "
                    f"{got:+.6f} (want {float(want):+.6f})"
                )
            else:
                failures += 1
                print(
                    f"[FAIL] {task:11s} {smiles:40s} "
                    f"{got:+.6f} (want {float(want):+.6f}, "
                    f"off by {delta:.6g})"
                )

    print("=" * 72)
    if failures:
        print(
            f"{failures} of {checked} predictions do not match. Check the model "
            f"hashes, package versions, and source featurization before using them."
        )
        return 1
    print(f"All {checked} predictions match within {args.tolerance}. Gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
