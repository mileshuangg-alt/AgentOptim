"""Offline interactive 3D molecular viewer for Streamlit."""

from __future__ import annotations

import json


def _changed_atoms(parent, current) -> set[int]:
    """MCS-based atom changes, plus surviving attachment sites for deletions."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    if parent is None:
        return set()
    result = rdFMCS.FindMCS(
        [parent, current],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=2,
    )
    pattern = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    if pattern is None:
        return set(range(current.GetNumAtoms()))
    parent_match = parent.GetSubstructMatch(pattern)
    current_match = current.GetSubstructMatch(pattern)
    current_common = set(current_match)
    changed = set(range(current.GetNumAtoms())) - current_common
    parent_to_current = dict(zip(parent_match, current_match))
    parent_common = set(parent_match)
    for atom in parent.GetAtoms():
        if atom.GetIdx() in parent_common:
            continue
        for neighbour in atom.GetNeighbors():
            mapped = parent_to_current.get(neighbour.GetIdx())
            if mapped is not None:
                changed.add(mapped)
    return changed


def molecule_payload(smiles: str, parent_smiles: str | None = None) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor

    molecule = Chem.MolFromSmiles(smiles)
    parent = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    if molecule is None or (parent_smiles and parent is None):
        raise ValueError("invalid SMILES supplied to the 3D viewer")
    changed = _changed_atoms(parent, molecule)

    embedded = Chem.AddHs(Chem.Mol(molecule))
    status = AllChem.EmbedMolecule(embedded, randomSeed=0xA11CE)
    if status == 0:
        try:
            AllChem.MMFFOptimizeMolecule(embedded, maxIters=200)
        except Exception:
            pass
        embedded = Chem.RemoveHs(embedded)
        conformer = embedded.GetConformer()
    else:
        embedded = Chem.Mol(molecule)
        rdDepictor.Compute2DCoords(embedded)
        conformer = embedded.GetConformer()

    atoms = []
    for atom in embedded.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "element": atom.GetSymbol(),
                "x": round(position.x, 4),
                "y": round(position.y, 4),
                "z": round(position.z, 4),
                "changed": atom.GetIdx() in changed,
            }
        )
    bonds = [
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), float(bond.GetBondTypeAsDouble())]
        for bond in embedded.GetBonds()
    ]
    return {"atoms": atoms, "bonds": bonds, "changed_count": len(changed)}


def viewer_html(smiles: str, parent_smiles: str | None = None, height: int = 380) -> str:
    payload = json.dumps(molecule_payload(smiles, parent_smiles))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}body{{margin:0;font:12px system-ui;color:#dfe9e3;background:#122019}}
.head{{display:flex;justify-content:space-between;padding:9px 13px;color:#a9b9b0}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;background:#f0a328;margin-right:5px}}
canvas{{display:block;width:100%;height:{height - 42}px;cursor:grab;background:radial-gradient(circle,#294036,#101a16)}}
canvas:active{{cursor:grabbing}}
</style></head><body><div class="head"><span>Drag to rotate · scroll to zoom</span>
<span class="legend"><i></i> changed atom or attachment site</span></div>
<canvas id="view"></canvas><script>
const data={payload}, colours={{C:"#9aa5a0",N:"#4a84d8",O:"#e6534d",F:"#75c96b",Cl:"#45a85e",S:"#e1c34f",P:"#e99043"}};
const canvas=document.getElementById("view"),state={{rx:-.35,ry:.45,zoom:52,drag:false,x:0,y:0}};
function draw(){{
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
 const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
 const pts=data.atoms.map(a=>{{let x=a.x,y=a.y,z=a.z,cy=Math.cos(state.ry),sy=Math.sin(state.ry),cx=Math.cos(state.rx),sx=Math.sin(state.rx);
  [x,z]=[x*cy+z*sy,-x*sy+z*cy];[y,z]=[y*cx-z*sx,y*sx+z*cx];
  return {{x:rect.width/2+x*state.zoom,y:rect.height/2+y*state.zoom,z}}}});
 ctx.lineWidth=3;ctx.strokeStyle="#91a59a";
 data.bonds.forEach(b=>{{ctx.beginPath();ctx.moveTo(pts[b[0]].x,pts[b[0]].y);ctx.lineTo(pts[b[1]].x,pts[b[1]].y);ctx.stroke()}});
 [...data.atoms.keys()].sort((a,b)=>pts[a].z-pts[b].z).forEach(i=>{{const atom=data.atoms[i],p=pts[i],r=atom.changed?10:8;
  ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=atom.changed?"#f0a328":(colours[atom.element]||"#aab3af");ctx.fill();
  if(atom.changed){{ctx.strokeStyle="#ffe1a0";ctx.lineWidth=2;ctx.stroke()}}
  if(atom.element!=="C"){{ctx.fillStyle="white";ctx.font="9px system-ui";ctx.textAlign="center";ctx.fillText(atom.element,p.x,p.y+3)}}}});
}}
canvas.onpointerdown=e=>{{state.drag=true;state.x=e.clientX;state.y=e.clientY;canvas.setPointerCapture(e.pointerId)}};
canvas.onpointermove=e=>{{if(!state.drag)return;state.ry+=(e.clientX-state.x)*.012;state.rx+=(e.clientY-state.y)*.012;state.x=e.clientX;state.y=e.clientY;draw()}};
canvas.onpointerup=()=>state.drag=false;canvas.onpointercancel=()=>state.drag=false;
canvas.onwheel=e=>{{e.preventDefault();state.zoom=Math.max(18,Math.min(100,state.zoom*(e.deltaY>0?.9:1.1)));draw()}};
window.onresize=draw;draw();
</script></body></html>"""
