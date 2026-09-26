"""Offline interactive 3D molecular viewer for Streamlit."""

from __future__ import annotations

import json


def _conformer(molecule):
    """Generate and minimize several 3D conformers, returning the lowest energy."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor, rdMolTransforms

    embedded = Chem.AddHs(Chem.Mol(molecule))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0xA11CE
    parameters.pruneRmsThresh = 0.25
    parameters.useSmallRingTorsions = True
    parameters.useMacrocycleTorsions = True
    conformer_count = 8 if embedded.GetNumAtoms() <= 80 else 4
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            embedded,
            numConfs=conformer_count,
            params=parameters,
        )
    )

    if not conformer_ids:
        rdDepictor.Compute2DCoords(embedded)
        return embedded, embedded.GetConformer().GetId(), "2D coordinate fallback"

    best_id = conformer_ids[0]
    method = "ETKDGv3"
    try:
        if AllChem.MMFFHasAllMoleculeParams(embedded):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                embedded,
                numThreads=0,
                maxIters=500,
                mmffVariant="MMFF94s",
            )
            best_id = min(
                zip(conformer_ids, results),
                key=lambda item: item[1][1],
            )[0]
            method = "ETKDGv3 + MMFF94s"
        elif AllChem.UFFHasAllMoleculeParams(embedded):
            results = AllChem.UFFOptimizeMoleculeConfs(
                embedded,
                numThreads=0,
                maxIters=500,
            )
            best_id = min(
                zip(conformer_ids, results),
                key=lambda item: item[1][1],
            )[0]
            method = "ETKDGv3 + UFF"
    except (RuntimeError, ValueError):
        # ETKDG still supplies valid 3D coordinates when a force field cannot
        # parameterize an unusual atom or charge state.
        method = "ETKDGv3"

    rdMolTransforms.CanonicalizeConformer(embedded.GetConformer(best_id))
    return embedded, best_id, method


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
    from rdkit.Chem import rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    parent = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    if molecule is None or (parent_smiles and parent is None):
        raise ValueError("invalid SMILES supplied to the 3D viewer")
    changed = _changed_atoms(parent, molecule)

    embedded, conformer_id, geometry = _conformer(molecule)
    conformer = embedded.GetConformer(conformer_id)
    positions = [
        conformer.GetAtomPosition(index)
        for index in range(embedded.GetNumAtoms())
    ]
    centre = (
        sum(position.x for position in positions) / len(positions),
        sum(position.y for position in positions) / len(positions),
        sum(position.z for position in positions) / len(positions),
    )
    periodic_table = Chem.GetPeriodicTable()

    atoms = []
    for atom in embedded.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "x": round(position.x - centre[0], 4),
                "y": round(position.y - centre[1], 4),
                "z": round(position.z - centre[2], 4),
                "radius": round(
                    periodic_table.GetRcovalent(atom.GetAtomicNum()), 3
                ),
                # Explicit hydrogens improve the model, but the edit highlight
                # stays on changed heavy atoms so one substitution cannot make
                # the whole molecule appear modified.
                "changed": atom.GetAtomicNum() > 1 and atom.GetIdx() in changed,
            }
        )
    bonds = [
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), float(bond.GetBondTypeAsDouble())]
        for bond in embedded.GetBonds()
    ]
    return {
        "atoms": atoms,
        "bonds": bonds,
        "changed_count": sum(atom["changed"] for atom in atoms),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "geometry": geometry,
    }


def viewer_html(smiles: str, parent_smiles: str | None = None, height: int = 380) -> str:
    payload = json.dumps(molecule_payload(smiles, parent_smiles))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}body{{margin:0;font:12px system-ui;color:#dfe9e3;background:#122019}}
.head{{display:flex;justify-content:space-between;gap:12px;padding:9px 13px;color:#a9b9b0}}
.geometry{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.legend{{white-space:nowrap}}.legend i{{display:inline-block;width:11px;height:11px;
 border:2px solid #f0a328;border-radius:50%;background:#9aa5a0;margin-right:5px;vertical-align:-2px}}
canvas{{display:block;width:100%;height:{height - 42}px;cursor:grab;background:radial-gradient(circle,#294036,#101a16)}}
canvas:active{{cursor:grabbing}}
</style></head><body><div class="head"><span class="geometry" id="geometry"></span>
<span class="legend"><i></i> changed atom · drag to rotate · scroll to zoom</span></div>
<canvas id="view"></canvas><script>
const data={payload};
const colours={{H:"#f4f5f2",C:"#707976",N:"#315fce",O:"#d93b36",F:"#52b84f",
 Cl:"#3aaa55",Br:"#8b3f2f",I:"#6f45a5",S:"#e4bd23",P:"#e67831",B:"#dc8a72"}};
const canvas=document.getElementById("view");
const initial={{rx:-.35,ry:.55}},state={{...initial,zoom:null,drag:false,x:0,y:0}};
document.getElementById("geometry").textContent=
 `${{data.formula}} · ${{data.geometry}} · explicit H`;
function shade(hex,amount){{
 const value=parseInt(hex.slice(1),16),target=amount<0?0:255,weight=Math.abs(amount)/100;
 const r=value>>16,g=(value>>8)&255,b=value&255;
 return `rgb(${{Math.round(r+(target-r)*weight)}},${{Math.round(g+(target-g)*weight)}},${{Math.round(b+(target-b)*weight)}})`;
}}
function line(ctx,a,b,colour,width,dash=[]){{
 ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
 ctx.strokeStyle=colour;ctx.lineWidth=width;ctx.lineCap="round";ctx.stroke();ctx.setLineDash([]);
}}
function drawBond(ctx,bond,pts){{
 const a=pts[bond[0]],b=pts[bond[1]],order=bond[2],dx=b.x-a.x,dy=b.y-a.y;
 const length=Math.hypot(dx,dy)||1,nx=-dy/length,ny=dx/length;
 let offsets=[0],dash=[];
 if(order>=2.8)offsets=[-3.2,0,3.2];
 else if(order>=1.8)offsets=[-2.3,2.3];
 else if(order>1.2){{offsets=[-2,2];dash=[[],[3,3]]}}
 const midpoint={{x:(a.x+b.x)/2,y:(a.y+b.y)/2}};
 offsets.forEach((offset,index)=>{{
  const shift={{x:nx*offset,y:ny*offset}};
  const start={{x:a.x+shift.x,y:a.y+shift.y}};
  const middle={{x:midpoint.x+shift.x,y:midpoint.y+shift.y}};
  const end={{x:b.x+shift.x,y:b.y+shift.y}};
  const width=Math.max(1.5,3.8*(a.scale+b.scale)/2);
  const pattern=Array.isArray(dash[index])?dash[index]:[];
  line(ctx,start,middle,colours[data.atoms[bond[0]].element]||"#aeb7b2",width,pattern);
  line(ctx,middle,end,colours[data.atoms[bond[1]].element]||"#aeb7b2",width,pattern);
 }});
}}
function drawAtom(ctx,atom,p){{
 const base=colours[atom.element]||"#b1aaa3";
 const radius=Math.max(4.2,atom.radius*9.5)*p.scale;
 if(atom.changed){{
  ctx.beginPath();ctx.arc(p.x,p.y,radius+5,0,Math.PI*2);
  ctx.fillStyle="rgba(240,163,40,.20)";ctx.fill();
  ctx.strokeStyle="#f0a328";ctx.lineWidth=2.5;ctx.stroke();
 }}
 ctx.save();ctx.shadowColor="rgba(0,0,0,.45)";ctx.shadowBlur=4;ctx.shadowOffsetY=2;
 const gradient=ctx.createRadialGradient(
  p.x-radius*.38,p.y-radius*.42,Math.max(1,radius*.08),p.x,p.y,radius
 );
 gradient.addColorStop(0,shade(base,62));gradient.addColorStop(.38,base);
 gradient.addColorStop(1,shade(base,-48));
 ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fillStyle=gradient;ctx.fill();
 ctx.restore();
 ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);
 ctx.strokeStyle=atom.element==="H"?"rgba(90,100,95,.7)":"rgba(255,255,255,.20)";
 ctx.lineWidth=.8;ctx.stroke();
 if(atom.element!=="C"&&atom.element!=="H"){{
  ctx.fillStyle=atom.element==="S"?"#322b15":"white";
  ctx.font=`600 ${{Math.max(8,8.5*p.scale)}}px system-ui`;ctx.textAlign="center";
  ctx.textBaseline="middle";ctx.fillText(atom.element,p.x,p.y+.4);
 }}
}}
function draw(){{
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
 const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
 const pts=data.atoms.map(a=>{{let x=a.x,y=a.y,z=a.z,cy=Math.cos(state.ry),sy=Math.sin(state.ry),cx=Math.cos(state.rx),sx=Math.sin(state.rx);
  [x,z]=[x*cy+z*sy,-x*sy+z*cy];[y,z]=[y*cx-z*sx,y*sx+z*cx];
  return {{rx:x,ry:y,z}}}});
 if(state.zoom===null){{
  const xs=pts.map(p=>p.rx),ys=pts.map(p=>p.ry);
  const span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),1);
  state.zoom=Math.max(14,Math.min(68,Math.min(rect.width,rect.height)*.72/span));
 }}
 pts.forEach(p=>{{
  p.scale=Math.max(.78,Math.min(1.22,1+p.z*.028));
  p.x=rect.width/2+p.rx*state.zoom*p.scale;
  p.y=rect.height/2+p.ry*state.zoom*p.scale;
 }});
 [...data.bonds].sort((a,b)=>
  (pts[a[0]].z+pts[a[1]].z)-(pts[b[0]].z+pts[b[1]].z)
 ).forEach(bond=>drawBond(ctx,bond,pts));
 [...data.atoms.keys()].sort((a,b)=>pts[a].z-pts[b].z)
  .forEach(index=>drawAtom(ctx,data.atoms[index],pts[index]));
}}
canvas.onpointerdown=e=>{{state.drag=true;state.x=e.clientX;state.y=e.clientY;canvas.setPointerCapture(e.pointerId)}};
canvas.onpointermove=e=>{{if(!state.drag)return;state.ry+=(e.clientX-state.x)*.012;state.rx+=(e.clientY-state.y)*.012;state.x=e.clientX;state.y=e.clientY;draw()}};
canvas.onpointerup=()=>state.drag=false;canvas.onpointercancel=()=>state.drag=false;
canvas.onwheel=e=>{{e.preventDefault();state.zoom=Math.max(10,Math.min(110,state.zoom*(e.deltaY>0?.9:1.1)));draw()}};
canvas.ondblclick=()=>{{state.rx=initial.rx;state.ry=initial.ry;state.zoom=null;draw()}};
window.onresize=()=>{{state.zoom=null;draw()}};draw();
</script></body></html>"""
