"""ADMET property backends: the inherited XGBoost models, or a labelled surrogate.

Two backends implement the same three-method interface
(`solubility`, `bbb`, `herg`):

  XGBoostBackend    the inherited models in `models/`. Real predictions.
  SurrogateBackend  transparent physchem heuristics. NOT a trained model.

Every score record says which backend produced it, and the frontend prints
that label. A surrogate number wearing an "0.92 ROC-AUC" claim in front of
judges is the one failure mode of this build that is dishonest rather than
merely broken.

FEATURIZATION -- mirrored from the source repo's `app.py:featurize_one`
(tuhinc5203/admet-property-prediction), which is itself lifted from
`Model_Improvement.ipynb`:

  1. strip salts by keeping the largest fragment
  2. 2048-bit Morgan fingerprint, radius 2, as bits (not counts)
  3. six RDKit descriptors in THIS ORDER:
         MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds
  4. assembled into a pandas DataFrame with columns fp_0..fp_2047 then the
     six descriptor names

Step 3's order is not the obvious one -- TPSA sits third, not last -- and step 4
is what makes a mistake survivable. The saved estimators carry
`feature_names_in_`, so a DataFrame with wrong column names raises. A bare
numpy array of the right width does not: it would be silently scored against
mismatched columns and every number downstream would be void. That is why this
builds a named DataFrame and why `scripts/verify_models.py` diffs this function
against the source repo's own.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(
    os.environ.get("ADMET_MODELS_DIR", str(_REPO_ROOT / "models"))
).expanduser()

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Order is load-bearing. See the module docstring.
DESCRIPTOR_ORDER = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                    "NumRotatableBonds")

FEATURE_NAMES = [f"fp_{i}" for i in range(MORGAN_BITS)] + list(DESCRIPTOR_ORDER)

# Filenames as the source repo saves them, plus a tolerated pickle alias.
_MODEL_FILES = {
    "solubility": ("solubility_xgb.pkl", "solubility.pkl"),
    "bbb": ("bbb_xgb.pkl", "bbb.pkl"),
    "herg": ("herg_xgb.pkl", "herg.pkl"),
}

# Reported by the source repo's README, on TDC scaffold splits.
SOURCE_METRICS = {
    "solubility": {"task": "regression", "metric": "MAE", "value": 0.899,
                   "tdc_sota": 0.741},
    "bbb": {"task": "classification", "metric": "ROC-AUC", "value": 0.920,
            "tdc_sota": 0.916},
    "herg": {"task": "classification", "metric": "ROC-AUC", "value": 0.856,
             "tdc_sota": 0.880},
}

_REGRESSION_TASKS = {"solubility"}


def largest_fragment(mol):
    """Keep the biggest fragment, dropping salts and counter-ions."""
    from rdkit import Chem

    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def featurize(smiles: str):
    """Named 1x2054 DataFrame, matching the source repo's pipeline exactly.

    Set ADMET_FEATURIZER="package.module:function" to delegate to the source
    repo's own function instead -- the documented fix if `verify_models.py`
    ever reports a divergence.
    """
    override = os.environ.get("ADMET_FEATURIZER")
    if override:
        module_name, _, func_name = override.partition(":")
        func = getattr(importlib.import_module(module_name), func_name)
        return func(smiles)

    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = largest_fragment(mol)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS
    )
    bits = np.zeros((MORGAN_BITS,), dtype=int)
    ConvertToNumpyArray(fp, bits)

    row = {f"fp_{i}": int(bit) for i, bit in enumerate(bits)}
    for name in DESCRIPTOR_ORDER:
        # Preserve RDKit's native result type. The source app keeps the three
        # count descriptors as integers and the continuous descriptors as
        # floats, and the verification gate checks exact DataFrame equality.
        row[name] = getattr(Descriptors, name)(mol)
    return pd.DataFrame([row], columns=FEATURE_NAMES)


class SurrogateBackend:
    """Physchem heuristics standing in for the trained ADMET models.

    Directionally sane and monotone in the properties medicinal chemists
    actually trade off -- polarity raises solubility and lowers BBB; lipophilic
    bases with long tails raise hERG risk -- which is enough to make the agents
    genuinely disagree. It is not calibrated and carries no accuracy claim.
    """

    name = "surrogate"
    is_surrogate = True
    metrics: dict = {}

    def _props(self, smiles: str) -> dict:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return {
            "logp": Crippen.MolLogP(mol),
            "mw": Descriptors.MolWt(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "arom": rdMolDescriptors.CalcNumAromaticRings(mol),
            "basic_n": sum(
                1
                for a in mol.GetAtoms()
                if a.GetSymbol() == "N"
                and not a.GetIsAromatic()
                and a.GetTotalNumHs() + a.GetDegree() <= 3
            ),
        }

    def solubility(self, smiles: str) -> float:
        """logS, in the spirit of the ESOL equation (Delaney 2004)."""
        p = self._props(smiles)
        logs = 0.16 - 0.63 * p["logp"] - 0.0062 * p["mw"] + 0.066 * p["rotb"] - 0.74 * p["arom"]
        return max(-12.0, min(1.5, logs))

    def bbb(self, smiles: str) -> float:
        """P(penetrant). Rewards moderate lipophilicity, punishes polarity."""
        p = self._props(smiles)
        z = (
            1.4
            + 0.55 * min(p["logp"], 5.0)
            - 0.035 * p["tpsa"]
            - 0.004 * max(0.0, p["mw"] - 400.0)
            - 0.45 * p["hbd"]
        )
        return 1.0 / (1.0 + pow(2.718281828459045, -z))

    def herg(self, smiles: str) -> float:
        """P(blockade). The classic pharmacophore: lipophilic + basic amine."""
        p = self._props(smiles)
        z = (
            -3.4
            + 0.62 * p["logp"]
            + 0.85 * min(p["basic_n"], 2)
            + 0.10 * p["arom"]
            + 0.006 * max(0.0, p["mw"] - 250.0)
            - 0.018 * p["tpsa"]
        )
        return 1.0 / (1.0 + pow(2.718281828459045, -z))


class XGBoostBackend:
    """The inherited XGBoost models, loaded from `models/`.

    Constructing this raises unless all three load, so a half-wired backend can
    never serve surrogate numbers alongside real ones.
    """

    name = "xgboost"
    is_surrogate = False

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._models = {task: self._load(task) for task in _MODEL_FILES}
        self.metrics = dict(SOURCE_METRICS)
        override = self.models_dir / "metrics.json"
        if override.exists():
            self.metrics.update(json.loads(override.read_text()))

    def _load(self, task: str):
        import joblib

        for filename in _MODEL_FILES[task]:
            path = self.models_dir / filename
            if path.exists():
                return joblib.load(path)
        raise FileNotFoundError(
            f"no model for {task!r} in {self.models_dir} "
            f"(looked for {', '.join(_MODEL_FILES[task])})"
        )

    def _predict(self, task: str, smiles: str) -> float:
        model = self._models[task]
        features = featurize(smiles)
        if task in _REGRESSION_TASKS:
            return float(model.predict(features)[0])
        return float(model.predict_proba(features)[0][1])

    def solubility(self, smiles: str) -> float:
        """Predicted logS (log mol/L). Source MAE 0.899 on a TDC scaffold split."""
        return self._predict("solubility", smiles)

    def bbb(self, smiles: str) -> float:
        """P(BBB penetrant). Source ROC-AUC 0.920."""
        return self._predict("bbb", smiles)

    def herg(self, smiles: str) -> float:
        """P(hERG blockade). Source ROC-AUC 0.856."""
        return self._predict("herg", smiles)


def get_backend(prefer_real: bool = True):
    """Return the XGBoost backend when the models are present, else surrogate.

    Set ADMET_BACKEND=surrogate to force the surrogate, or
    ADMET_BACKEND=xgboost to fail loudly rather than fall back.
    """
    forced = os.environ.get("ADMET_BACKEND", "").strip().lower()
    if forced == "surrogate":
        return SurrogateBackend()
    if forced == "xgboost":
        return XGBoostBackend()
    if prefer_real:
        try:
            return XGBoostBackend()
        except Exception:
            pass
    return SurrogateBackend()
