"""`propose()`: the edit engine.

Two tiers, in the order the plan builds them:

  mutate()       deterministic SMARTS transforms. No network, no API key.
  llm_propose()  an LLM emits edited SMILES given the parent and the round's
                 agent notes. Everything it returns is validated; invalids are
                 dropped silently and the shortfall is topped up from mutate().

`propose()` never returns empty and never returns an invalid SMILES. When the
LLM is unavailable it degrades to pure mutation without comment, which is the
fourth item on the cut list already taken.

Each transform is tagged with the axis it is *meant* to push, so the agent
transcript can say "fluorinate to block metabolism" rather than printing a
SMARTS string at the judges.
"""

from __future__ import annotations

import os
import random

from core.contract import canonical, is_valid

# (name, SMARTS reaction, intent, axis it targets)
TRANSFORMS = (
    ("aryl_fluorination", "[cH:1]>>[c:1]F",
     "block aromatic metabolism, trim electron density", "affinity"),
    ("aryl_chlorination", "[cH:1]>>[c:1]Cl",
     "add lipophilic bulk into a hydrophobic pocket", "affinity"),
    ("aryl_methylation", "[cH:1]>>[c:1]C",
     "fill a small pocket, raise lipophilicity", "affinity"),
    ("aryl_methoxylation", "[cH:1]>>[c:1]OC",
     "add an H-bond acceptor without much polarity cost", "bbb"),
    ("aryl_hydroxylation", "[cH:1]>>[c:1]O",
     "add a donor to raise solubility", "solubility"),
    ("aryl_nitrile", "[cH:1]>>[c:1]C#N",
     "polar, small, metabolically robust acceptor", "solubility"),
    ("aryl_carboxamide", "[cH:1]>>[c:1]C(N)=O",
     "primary amide: strong solubility gain, BBB cost", "solubility"),
    ("aryl_carboxylic_acid", "[cH:1]>>[c:1]C(O)=O",
     "acid: large solubility gain, usually kills BBB", "solubility"),
    ("aryl_sulfonamide", "[cH:1]>>[c:1]S(N)(=O)=O",
     "polar acceptor, lowers hERG risk", "herg"),
    ("phenyl_to_pyridyl", "[cH:1]:[cH:2]>>[n:1]:[cH:2]",
     "swap CH for N: cuts logP and hERG liability", "herg"),
    ("methyl_to_ethyl", "[CH3:1][#6:2]>>[CH2:1](C)[#6:2]",
     "homologate to probe a deeper pocket", "affinity"),
    ("methyl_to_trifluoromethyl", "[CH3:1][c:2]>>[C:1](F)(F)F[c:2]",
     "CF3: metabolically blocked, more lipophilic", "affinity"),
    ("hydroxyl_to_methyl_ether", "[OX2H:1][#6:2]>>[O:1](C)[#6:2]",
     "cap a donor to regain membrane permeability", "bbb"),
    ("amine_n_methylation", "[NX3;H1;!$(NC=O):1]>>[N:1]C",
     "N-methylate: fewer donors, better BBB", "bbb"),
    ("halide_to_fluoride", "[c:1][Cl,Br,I]>>[c:1]F",
     "downsize the halogen to cut logP and hERG risk", "herg"),
    ("defluorinate", "[c:1]F>>[cH:1]",
     "strip a fluorine to reduce lipophilicity", "herg"),
    ("ketone_to_alcohol", "[CX3:1](=[OX1])[#6:2]>>[CX4:1](O)[#6:2]",
     "reduce the carbonyl: more polar, more flexible", "solubility"),
    ("add_morpholine_tail", "[cH:1]>>[c:1]N1CCOCC1",
     "morpholine: the standard solubilising handle", "solubility"),
    ("piperidine_to_piperazine", "[CH2:1][NX3:2]([CH2:3])>>[CH2:1][N:2]([CH2:3])",
     "keep the basic centre, open a vector", "affinity"),
)

_MAX_HEAVY_ATOMS = 55  # stop the loop growing a molecule into a polymer


def _apply(transform_smarts: str, parent_mol) -> list[str]:
    """All single-site products of one transform, as canonical SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        rxn = AllChem.ReactionFromSmarts(transform_smarts)
    except Exception:
        return []
    if rxn is None:
        return []

    products: list[str] = []
    try:
        outcomes = rxn.RunReactants((parent_mol,))
    except Exception:
        return []

    for outcome in outcomes:
        for product in outcome:
            try:
                Chem.SanitizeMol(product)
                smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            if not smiles or not is_valid(smiles):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
                continue
            products.append(Chem.MolToSmiles(mol))
    return products


def mutate(parent: str, n: int = 6, seed: int | None = None,
           prefer_axes: tuple[str, ...] = ()) -> list[str]:
    """Up to `n` distinct valid children of `parent` by SMARTS transform.

    `prefer_axes` biases which transforms are tried first, so the agents'
    stated priorities steer the chemistry instead of only judging it.
    Deterministic for a given `seed`.
    """
    from rdkit import Chem

    parent_mol = Chem.MolFromSmiles(parent)
    if parent_mol is None:
        raise ValueError(f"invalid parent SMILES: {parent!r}")
    parent_canonical = Chem.MolToSmiles(parent_mol)

    rng = random.Random(seed)
    ordered = list(TRANSFORMS)
    rng.shuffle(ordered)
    if prefer_axes:
        ordered.sort(key=lambda t: t[3] not in prefer_axes)

    children: dict[str, tuple[str, str]] = {}
    for name, smarts, intent, _axis in ordered:
        options = _apply(smarts, parent_mol)
        rng.shuffle(options)
        for smiles in options:
            if smiles == parent_canonical or smiles in children:
                continue
            children[smiles] = (name, intent)
            break  # one product per transform keeps the batch diverse
        if len(children) >= n:
            break

    MUTATION_LOG.clear()
    MUTATION_LOG.update(children)
    return list(children)


# Populated by the most recent `mutate()` call: child SMILES -> (transform, intent).
# The transcript reads this so each candidate can be shown with its rationale.
MUTATION_LOG: dict[str, tuple[str, str]] = {}


def _llm_propose(parent: str, context: dict, n: int) -> list[str]:
    """Ask an LLM for edited SMILES. Returns [] on any failure."""
    from core.llm import complete_json

    notes = "\n".join(f"- {note}" for note in context.get("agent_notes", []) or [])
    scores = context.get("scores") or {}
    prompt = (
        f"Parent molecule (SMILES): {parent}\n"
        f"Target or phenotype: {context.get('target', 'not supplied')}\n"
        f"Why this target: {context.get('target_rationale', 'not supplied')}\n"
        f"Program objective: {context.get('objective', 'not supplied')}\n"
        f"Activity evidence: {context.get('activity_strategy', 'not supplied')}\n"
        f"BBB exposure goal: {context.get('bbb_goal', 'penetrate')}\n"
        f"Round: {context.get('round', 1)}\n"
        f"Current normalised scores (0-1, higher better): {scores}\n"
        f"Specialist agent notes from the previous round:\n{notes or '- none yet'}\n\n"
        f"Propose {n} single- or double-step medicinal chemistry edits to the parent. "
        "Keep the core scaffold recognisable. Address the weakest axes named above.\n"
        'Reply with JSON only: {"candidates": [{"smiles": "...", "rationale": "..."}]}'
    )
    payload = complete_json(
        system=(
            "You are a medicinal chemist proposing analogues for lead optimisation. "
            "Emit only chemically valid SMILES for stable, synthesisable molecules. "
            "Do not restate the parent unchanged."
        ),
        user=prompt,
    )
    if not payload:
        return []

    out: list[str] = []
    for item in payload.get("candidates", []) or []:
        smiles = (item or {}).get("smiles") if isinstance(item, dict) else item
        if not isinstance(smiles, str) or not is_valid(smiles):
            continue  # drop invalids silently, per the contract
        try:
            smiles = canonical(smiles)
        except ValueError:
            continue
        if smiles == canonical(parent) or smiles in out:
            continue
        out.append(smiles)
        rationale = (item or {}).get("rationale") if isinstance(item, dict) else None
        MUTATION_LOG[smiles] = ("llm_edit", rationale or "LLM-proposed analogue")
    return out


def propose(parent: str, context: dict) -> list[str]:
    """3-8 valid, distinct children of `parent`. See `core.contract.propose`."""
    context = context or {}
    target = int(context.get("n_candidates", 6))
    target = max(3, min(8, target))
    prefer = tuple(context.get("prefer_axes") or ())
    seed = context.get("seed")

    candidates: list[str] = []
    if os.environ.get("EDIT_ENGINE", "").strip().lower() != "mutation":
        try:
            candidates = _llm_propose(parent, context, target)
        except Exception:
            candidates = []

    if len(candidates) < 3:
        log_backup = dict(MUTATION_LOG)
        topped_up = mutate(parent, n=target, seed=seed, prefer_axes=prefer)
        MUTATION_LOG.update(log_backup)
        for smiles in topped_up:
            if smiles not in candidates:
                candidates.append(smiles)

    if not candidates:
        # A fully substituted parent with no applicable transform. Widen the
        # net rather than hand the loop an empty list.
        candidates = mutate(parent, n=target, seed=(seed or 0) + 1)
    if not candidates:
        raise RuntimeError(f"no valid edits found for {parent!r}")

    return candidates[:8]
