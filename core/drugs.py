"""Scientist-facing compound lookup for supported and reference drugs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Drug:
    name: str
    aliases: tuple[str, ...]
    smiles: str
    target: str
    rationale: str
    objective: str
    optimizable: bool
    limitation: str = ""
    source: str = "Curated example"
    pubchem_cid: int | None = None
    bbb_goal: str = "penetrate"
    original_smiles: str = ""


HALOPERIDOL = Drug(
    name="Haloperidol",
    aliases=("Haldol",),
    smiles="O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    target="DRD2",
    rationale=(
        "DRD2 is a dopamine receptor central to antipsychotic activity. "
        "Haloperidol is a potent, clinically familiar starting scaffold."
    ),
    objective=(
        "Preserve DRD2 affinity and CNS exposure while improving solubility "
        "and hERG safety. Keep affinity above the programme floor."
    ),
    optimizable=True,
)

IBUPROFEN = Drug(
    name="Ibuprofen",
    aliases=("Advil", "Motrin"),
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    target="COX-1 / COX-2",
    rationale=(
        "COX-1 and COX-2 catalyse prostaglandin synthesis. Ibuprofen is a "
        "clinically familiar anti-inflammatory reference scaffold."
    ),
    objective=(
        "A COX programme would preserve enzyme inhibition while improving "
        "solubility, selectivity, safety, and synthetic accessibility."
    ),
    optimizable=True,
    limitation=(
        "No validated COX-1/COX-2 activity oracle is connected. Optimization "
        "uses similarity to ibuprofen only as a scaffold-retention proxy."
    ),
    bbb_goal="avoid",
)

DRUGS = (HALOPERIDOL, IBUPROFEN)


def search(query: str) -> list[Drug]:
    query = (query or "").strip().lower()
    if not query:
        return list(DRUGS)
    return [
        drug
        for drug in DRUGS
        if query in drug.name.lower()
        or any(query in alias.lower() for alias in drug.aliases)
    ]


def from_pubchem(compound) -> Drug:
    """Create an editable target program around a PubChem structure."""
    fragment_note = (
        " PubChem returned multiple fragments; the largest fragment is used "
        "for scoring and optimization."
        if compound.original_smiles
        and compound.original_smiles != compound.smiles
        else ""
    )
    return Drug(
        name=compound.name,
        aliases=(),
        smiles=compound.smiles,
        target="",
        rationale="",
        objective=(
            "Preserve the starting scaffold while improving solubility, hERG "
            "safety, and synthetic accessibility for the selected target program."
        ),
        optimizable=True,
        limitation=(
            "A target-specific activity oracle has not been connected for this "
            "compound. The run will use starting-structure similarity as an "
            f"explicitly labelled activity-retention proxy.{fragment_note}"
        ),
        source=f"PubChem CID {compound.cid}",
        pubchem_cid=compound.cid,
        bbb_goal="neutral",
        original_smiles=compound.original_smiles,
    )
