"""`score()`: the one function the whole loop reads properties through.

Assembles five axes from three sources -- a target-specific oracle or labelled
seed-similarity proxy, the ADMET backend, and RDKit's synthetic accessibility
score -- and stamps every record with the provenance of each number.

Results are cached per SMILES: the loop re-scores the incumbent every round
and the frontend re-reads cached runs, so an uncached `score()` would spend
the demo's time budget on molecules it has already seen.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

from core import admet
from core.contract import AXES, canonical

# Known DRD2 antagonists, used only by the similarity fallback below.
DRD2_REFERENCE_LIGANDS = (
    "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",          # haloperidol
    "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",      # risperidone
    "Clc1ccccc1N1CCN(CCCCN2C(=O)CCC2=O)CC1",                  # buspirone-like
    "CN1CCCC1Cc1c[nH]c2ccc(CS(=O)(=O)N)cc12",                 # sumatriptan-like
    "Cc1cc(C)c2c(c1)N(CCN1CCN(C)CC1)c1ccccc1S2",              # thioridazine-like
    "COc1ccc2[nH]cc(CCN3CCC(c4ccccc4)CC3)c2c1",               # generic aryl-piperidine
)


class TDCAffinityOracle:
    """PyTDC's pretrained DRD2 activity oracle (Olivecrona et al. ECFP6 SVM)."""

    name = "tdc-drd2"
    is_surrogate = False

    def __init__(self):
        from tdc import Oracle

        self._oracle = Oracle(name="DRD2")
        # Fail here rather than mid-demo if the pickle did not download.
        self._oracle("CCO")

    def __call__(self, smiles: str) -> float:
        value = self._oracle(smiles)
        if isinstance(value, list):
            value = value[0]
        return float(value)


class SimilarityAffinityOracle:
    """Ligand-based stand-in: max Tanimoto to a panel of known DRD2 ligands.

    A real similarity measurement, not a trained activity model. It moves in
    the right direction when an edit walks away from the DRD2 pharmacophore,
    which is what the affinity agent needs to have an argument -- but it will
    happily reward a molecule that is similar to haloperidol and inactive.
    Named `surrogate-similarity` in provenance for exactly that reason.
    """

    name = "surrogate-similarity"
    is_surrogate = True

    def __init__(self):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        self._refs = [
            self._gen.GetFingerprint(Chem.MolFromSmiles(s))
            for s in DRD2_REFERENCE_LIGANDS
            if Chem.MolFromSmiles(s) is not None
        ]

    def __call__(self, smiles: str) -> float:
        from rdkit import Chem
        from rdkit.DataStructs import BulkTanimotoSimilarity

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        sims = BulkTanimotoSimilarity(self._gen.GetFingerprint(mol), self._refs)
        # Deliberately unscaled. An earlier version stretched this by 1.6x to
        # fill the 0-1 range; because the seed is itself in the panel, that
        # clipped every close analogue to exactly 1.00 and flattened the axis
        # the affinity agent argues over. Raw Tanimoto keeps the resolution.
        return min(1.0, max(sims))


class SeedSimilarityAffinityOracle:
    """Tanimoto similarity to the selected starting compound.

    This is the target-retention proxy for programs that do not have a
    target-specific oracle. It must never be presented as measured or predicted
    affinity: it only answers whether an analogue retained the seed's local
    chemical features.
    """

    name = "seed-similarity"
    is_surrogate = True

    def __init__(self, seed: str):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(seed)
        if mol is None:
            raise ValueError(f"invalid seed SMILES: {seed!r}")
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        self._reference = self._gen.GetFingerprint(mol)

    def __call__(self, smiles: str) -> float:
        from rdkit import Chem
        from rdkit.DataStructs import TanimotoSimilarity

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return float(
            TanimotoSimilarity(self._gen.GetFingerprint(mol), self._reference)
        )


def _sa_scorer():
    """RDKit's sascorer, which ships in Contrib rather than on sys.path."""
    import sys

    from rdkit.Chem import RDConfig

    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_path not in sys.path:
        sys.path.append(sa_path)
    import sascorer

    return sascorer.calculateScore


@functools.lru_cache(maxsize=1)
def _sources():
    """Resolve the three backends once. Cached so the demo pays setup once."""
    forced = os.environ.get("AFFINITY_ORACLE", "").strip().lower()
    if forced == "similarity":
        affinity = SimilarityAffinityOracle()
    elif forced == "tdc":
        affinity = TDCAffinityOracle()
    else:
        try:
            affinity = TDCAffinityOracle()
        except Exception:
            affinity = SimilarityAffinityOracle()
    return affinity, admet.get_backend(), _sa_scorer()


@dataclass
class ScoringContext:
    """Run-local oracle bundle for one compound and target program."""

    affinity: object
    backend: object
    sa_score: object
    target: str
    activity_note: str
    _cache: dict[str, tuple] = field(default_factory=dict)

    def provenance(self) -> dict:
        return {
            "affinity": self.affinity.name,
            "affinity_target": self.target,
            "activity_note": self.activity_note,
            "solubility": self.backend.name,
            "bbb": self.backend.name,
            "herg": self.backend.name,
            "sa": "rdkit-sascorer",
            "any_surrogate": bool(
                self.affinity.is_surrogate or self.backend.is_surrogate
            ),
        }

    def score(self, smiles: str) -> dict:
        from rdkit import Chem

        key = canonical(smiles)
        if key not in self._cache:
            mol = Chem.MolFromSmiles(key)
            self._cache[key] = (
                ("affinity", float(self.affinity(key))),
                ("solubility", float(self.backend.solubility(key))),
                ("bbb", float(self.backend.bbb(key))),
                ("herg", float(self.backend.herg(key))),
                ("sa", float(self.sa_score(mol))),
            )
        record = dict(self._cache[key])
        record["provenance"] = self.provenance()
        return record


def for_program(seed: str, target: str) -> ScoringContext:
    """Select a scientifically honest activity strategy for a target program."""
    default_affinity, backend, sa_score = _sources()
    target = (target or "").strip()
    target_key = "".join(character for character in target.lower() if character.isalnum())
    is_drd2 = target_key in {
        "drd2",
        "d2",
        "dopamined2",
        "dopamined2receptor",
        "d2dopaminereceptor",
    }
    if is_drd2:
        affinity = default_affinity
        if affinity.is_surrogate:
            activity_note = (
                "DRD2 activity is represented by similarity to a reference panel of "
                "known DRD2 ligands. This is a structural proxy, not an affinity model."
            )
        else:
            activity_note = (
                "DRD2 activity is scored by the PyTDC DRD2 oracle; experimental "
                "binding still has to confirm target engagement."
            )
    else:
        affinity = SeedSimilarityAffinityOracle(seed)
        activity_note = (
            f"No validated {target or 'target-specific'} activity oracle is connected. "
            "The activity axis is Tanimoto similarity to the selected starting "
            "compound, used only to discourage scaffold drift."
        )
    return ScoringContext(affinity, backend, sa_score, target, activity_note)


def provenance(context: ScoringContext | None = None) -> dict:
    """Which backend stands behind each axis, and whether it is a surrogate."""
    if context is not None:
        return context.provenance()
    affinity, backend, _ = _sources()
    return {
        "affinity": affinity.name,
        "solubility": backend.name,
        "bbb": backend.name,
        "herg": backend.name,
        "sa": "rdkit-sascorer",
        "any_surrogate": bool(affinity.is_surrogate or backend.is_surrogate),
    }


@functools.lru_cache(maxsize=4096)
def _score_canonical(smiles: str) -> tuple:
    affinity, backend, sa_score = _sources()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return (
        ("affinity", affinity(smiles)),
        ("solubility", backend.solubility(smiles)),
        ("bbb", backend.bbb(smiles)),
        ("herg", backend.herg(smiles)),
        ("sa", float(sa_score(mol))),
    )


def score(smiles: str, context: ScoringContext | None = None) -> dict:
    """Five raw property values plus provenance. Raises ValueError if invalid."""
    if context is not None:
        return context.score(smiles)
    record = dict(_score_canonical(canonical(smiles)))
    assert set(record) == set(AXES)
    record["provenance"] = provenance()
    return record


def warm_up() -> dict:
    """Resolve backends and score one molecule. Call before a timed demo."""
    score("CCO")
    return provenance()
