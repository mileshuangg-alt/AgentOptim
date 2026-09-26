"""Small, dependency-free PubChem client for compound name lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

AUTOCOMPLETE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{query}/json"
)
PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "Title,CanonicalSMILES,IsomericSMILES,MolecularFormula,IUPACName/JSON"
)
USER_AGENT = "AgentOptim/1.0 (PubChem compound lookup)"


class PubChemError(RuntimeError):
    """A readable PubChem network or response failure."""


@dataclass(frozen=True)
class PubChemCompound:
    cid: int
    name: str
    smiles: str
    molecular_formula: str = ""
    iupac_name: str = ""
    original_smiles: str = ""

    @property
    def source_url(self) -> str:
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.cid}"


def _fetch_json(url: str, timeout: float = 8.0) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise PubChemError("PubChem did not find that compound.") from exc
        raise PubChemError(f"PubChem returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PubChemError(f"PubChem lookup failed: {exc}") from exc


def search_names(query: str, limit: int = 8) -> list[str]:
    """Return PubChem autocomplete names for a partial compound name."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    limit = max(1, min(int(limit), 20))
    url = AUTOCOMPLETE_URL.format(query=quote(query, safe=""))
    payload = _fetch_json(f"{url}?limit={limit}")
    names = payload.get("dictionary_terms", {}).get("compound", [])
    return [str(name) for name in names[:limit] if str(name).strip()]


def fetch_compound(name: str) -> PubChemCompound:
    """Resolve a PubChem name to its preferred structure and identifiers."""
    name = (name or "").strip()
    if not name:
        raise PubChemError("Enter a compound name.")
    payload = _fetch_json(PROPERTY_URL.format(name=quote(name, safe="")))
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        raise PubChemError("PubChem returned no structure for that name.")

    record = properties[0]
    smiles = (
        record.get("SMILES")
        or record.get("IsomericSMILES")
        or record.get("ConnectivitySMILES")
        or record.get("CanonicalSMILES")
    )
    if not smiles:
        raise PubChemError("PubChem returned the compound without a SMILES structure.")

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PubChemError("PubChem returned a structure RDKit could not parse.")
    original_smiles = Chem.MolToSmiles(mol)
    fragments = Chem.GetMolFrags(mol, asMols=True)
    active_fragment = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    try:
        smiles = Chem.MolToSmiles(active_fragment)
    except Exception as exc:
        raise PubChemError("PubChem returned a structure RDKit could not parse.") from exc

    return PubChemCompound(
        cid=int(record["CID"]),
        name=str(record.get("Title") or name),
        smiles=smiles,
        molecular_formula=str(record.get("MolecularFormula") or ""),
        iupac_name=str(record.get("IUPACName") or ""),
        original_smiles=original_smiles,
    )
