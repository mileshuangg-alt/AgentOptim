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
    optimizable=False,
    limitation=(
        "This repository has a DRD2 affinity oracle, not a COX-1/COX-2 oracle. "
        "Ibuprofen can be inspected, but optimization is disabled to avoid "
        "presenting DRD2 similarity as COX activity."
    ),
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
