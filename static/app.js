const keys=["affinity","solubility","bbb","herg","sa"];
let run,current=-1,unlocked=-1,drugs=[],selectedDrug=null;
const el=id=>document.getElementById(id);

const elementColors={H:"#f4f5f2",C:"#707976",N:"#315fce",O:"#d93b36",F:"#52b84f",Cl:"#3aaa55",Br:"#8b3f2f",I:"#6f45a5",S:"#e4bd23",P:"#e67831",B:"#dc8a72"};
const viewer={rx:-.25,ry:.35,zoom:null,atoms:[],bonds:[],drag:false,x:0,y:0};
function parseSmiles(smiles){
  const atoms=[],bonds=[],stack=[],rings={};let current=null,i=0;
  while(i<smiles.length){
    const ch=smiles[i];
    if(ch==="("){stack.push(current);i++;continue}
    if(ch===")"){current=stack.pop();i++;continue}
    if(/\d/.test(ch)){if(rings[ch]===undefined)rings[ch]=current;else{bonds.push([rings[ch],current]);delete rings[ch]}i++;continue}
    if("-=#+\\/@.".includes(ch)){if(ch===".")current=null;i++;continue}
    let token=null;
    if(ch==="["){const end=smiles.indexOf("]",i);const raw=smiles.slice(i+1,end);token=(raw.match(/Cl|Br|[A-Z][a-z]?|[cnops]/)||["C"])[0];i=end+1}
    else if(smiles.slice(i,i+2)==="Cl"||smiles.slice(i,i+2)==="Br"){token=smiles.slice(i,i+2);i+=2}
    else if(/[A-Zcnops]/.test(ch)){token=ch;i++}
    else{i++;continue}
    const element=token.length===1&&token===token.toLowerCase()?token.toUpperCase():token;
    const next=atoms.length;atoms.push({element});
    if(current!==null)bonds.push([current,next]);current=next;
  }
  const n=Math.max(atoms.length,1);
  atoms.forEach((atom,index)=>{
    const angle=index*2.15, radius=46+(index%3)*8;
    atom.x=Math.cos(angle)*radius;atom.y=Math.sin(angle)*radius;atom.z=(index-(n-1)/2)*8;
  });
  return {atoms,bonds};
}
function changedAtoms(parent,current,atoms){
  const oldGraph=parseSmiles(parent),a=oldGraph.atoms.map(x=>x.element),b=atoms.map(x=>x.element);
  const table=Array.from({length:a.length+1},()=>Array(b.length+1).fill(0));
  for(let i=a.length-1;i>=0;i--)for(let j=b.length-1;j>=0;j--)table[i][j]=a[i]===b[j]?1+table[i+1][j+1]:Math.max(table[i+1][j],table[i][j+1]);
  const parentToCurrent=new Map(),matchedCurrent=new Set();let i=0,j=0;
  while(i<a.length&&j<b.length){
    if(a[i]===b[j]){parentToCurrent.set(i,j);matchedCurrent.add(j);i++;j++}
    else if(table[i+1][j]>=table[i][j+1])i++;else j++;
  }
  const changed=new Set();
  b.forEach((_,index)=>{if(!matchedCurrent.has(index))changed.add(index)});
  const matchedParent=new Set(parentToCurrent.keys());
  oldGraph.bonds.forEach(([left,right])=>{
    if(!matchedParent.has(left)&&parentToCurrent.has(right))changed.add(parentToCurrent.get(right));
    if(!matchedParent.has(right)&&parentToCurrent.has(left))changed.add(parentToCurrent.get(left));
  });
  if(!changed.size&&parent!==current&&atoms.length)changed.add(0);
  return changed;
}
function set3D(smiles,parent=smiles,payload=null){
  if(payload?.atoms?.length){
    viewer.atoms=payload.atoms;viewer.bonds=payload.bonds;
    viewer.changed=new Set(payload.atoms.map((atom,index)=>atom.changed?index:null).filter(index=>index!==null));
  }else{
    const parsed=parseSmiles(smiles);viewer.atoms=parsed.atoms;viewer.bonds=parsed.bonds;
    viewer.changed=changedAtoms(parent,smiles,viewer.atoms);
  }
  viewer.zoom=null;draw3D();
}
function project(atom){
  let x=atom.x,y=atom.y,z=atom.z;
  let cy=Math.cos(viewer.ry),sy=Math.sin(viewer.ry),cx=Math.cos(viewer.rx),sx=Math.sin(viewer.rx);
  [x,z]=[x*cy+z*sy,-x*sy+z*cy];[y,z]=[y*cx-z*sx,y*sx+z*cx];
  return{rx:x,ry:y,z};
}
function shade(hex,amount){
  const value=parseInt(hex.slice(1),16),target=amount<0?0:255,weight=Math.abs(amount)/100;
  const r=value>>16,g=(value>>8)&255,b=value&255;
  return`rgb(${Math.round(r+(target-r)*weight)},${Math.round(g+(target-g)*weight)},${Math.round(b+(target-b)*weight)})`;
}
function line(ctx,a,b,color,width,dash=[]){
  ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap="round";ctx.stroke();ctx.setLineDash([]);
}
function drawBond(ctx,bond,points){
  const a=points[bond[0]],b=points[bond[1]];if(!a||!b)return;
  const order=bond[2]||1,dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy)||1,nx=-dy/length,ny=dx/length;
  let offsets=[0],dashes=[];
  if(order>=2.8)offsets=[-3.2,0,3.2];
  else if(order>=1.8)offsets=[-2.3,2.3];
  else if(order>1.2){offsets=[-2,2];dashes=[[],[3,3]]}
  const middle={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
  offsets.forEach((offset,index)=>{
    const shift={x:nx*offset,y:ny*offset};
    const start={x:a.x+shift.x,y:a.y+shift.y},mid={x:middle.x+shift.x,y:middle.y+shift.y},end={x:b.x+shift.x,y:b.y+shift.y};
    const width=Math.max(1.5,3.8*(a.scale+b.scale)/2),dash=dashes[index]||[];
    line(ctx,start,mid,elementColors[viewer.atoms[bond[0]].element]||"#aeb7b2",width,dash);
    line(ctx,mid,end,elementColors[viewer.atoms[bond[1]].element]||"#aeb7b2",width,dash);
  });
}
function drawAtom(ctx,atom,point,index){
  const base=elementColors[atom.element]||"#b1aaa3";
  const radius=Math.max(4.2,(atom.radius||.76)*9.5)*point.scale;
  if(viewer.changed?.has(index)){
    ctx.beginPath();ctx.arc(point.x,point.y,radius+5,0,Math.PI*2);
    ctx.fillStyle="rgba(240,163,40,.20)";ctx.fill();ctx.strokeStyle="#f0a328";ctx.lineWidth=2.5;ctx.stroke();
  }
  ctx.save();ctx.shadowColor="rgba(0,0,0,.45)";ctx.shadowBlur=4;ctx.shadowOffsetY=2;
  const gradient=ctx.createRadialGradient(point.x-radius*.38,point.y-radius*.42,Math.max(1,radius*.08),point.x,point.y,radius);
  gradient.addColorStop(0,shade(base,62));gradient.addColorStop(.38,base);gradient.addColorStop(1,shade(base,-48));
  ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.fillStyle=gradient;ctx.fill();ctx.restore();
  ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);
  ctx.strokeStyle=atom.element==="H"?"rgba(90,100,95,.7)":"rgba(255,255,255,.20)";ctx.lineWidth=.8;ctx.stroke();
  if(atom.element!=="C"&&atom.element!=="H"){
    ctx.fillStyle=atom.element==="S"?"#322b15":"white";ctx.font=`600 ${Math.max(8,8.5*point.scale)}px sans-serif`;
    ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(atom.element,point.x,point.y+.4);
  }
}
function draw3D(){
  const canvas=el("molecule3d");if(!canvas)return;const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
  const points=viewer.atoms.map(project);
  if(viewer.zoom===null&&points.length){
    const xs=points.map(point=>point.rx),ys=points.map(point=>point.ry);
    const span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),1);
    viewer.zoom=Math.max(.5,Math.min(68,Math.min(rect.width,rect.height)*.72/span));
  }
  points.forEach(point=>{
    point.scale=Math.max(.78,Math.min(1.22,1+point.z*.028));
    point.x=rect.width/2+point.rx*viewer.zoom*point.scale;point.y=rect.height/2+point.ry*viewer.zoom*point.scale;
  });
  [...viewer.bonds].sort((a,b)=>(points[a[0]].z+points[a[1]].z)-(points[b[0]].z+points[b[1]].z)).forEach(bond=>drawBond(ctx,bond,points));
  [...viewer.atoms.keys()].sort((a,b)=>points[a].z-points[b].z).forEach(index=>drawAtom(ctx,viewer.atoms[index],points[index],index));
}
function setup3DControls(){
  const canvas=el("molecule3d");
  canvas.onpointerdown=e=>{viewer.drag=true;viewer.x=e.clientX;viewer.y=e.clientY;canvas.setPointerCapture(e.pointerId)};
  canvas.onpointermove=e=>{if(!viewer.drag)return;viewer.ry+=(e.clientX-viewer.x)*.012;viewer.rx+=(e.clientY-viewer.y)*.012;viewer.x=e.clientX;viewer.y=e.clientY;draw3D()};
  canvas.onpointerup=()=>viewer.drag=false;canvas.onpointercancel=()=>viewer.drag=false;
  canvas.onwheel=e=>{e.preventDefault();viewer.zoom=Math.max(.25,Math.min(110,viewer.zoom*(e.deltaY>0?.9:1.1)));draw3D()};
  canvas.ondblclick=()=>{viewer.rx=-.25;viewer.ry=.35;viewer.zoom=null;draw3D()};
  window.addEventListener("resize",()=>{viewer.zoom=null;draw3D()});
}

function moleculeSketch(smiles){
  const box=el("molecule"); box.innerHTML="";
  const n=8, cx=75, cy=75, r=50;
  const pts=Array.from({length:n},(_,i)=>[cx+r*Math.cos(i*Math.PI*2/n),cy+r*Math.sin(i*Math.PI*2/n)]);
  pts.forEach((p,i)=>{
    const q=pts[(i+1)%n],dx=q[0]-p[0],dy=q[1]-p[1],len=Math.hypot(dx,dy);
    const bond=document.createElement("i");bond.className="bond";
    Object.assign(bond.style,{left:p[0]+"px",top:p[1]+"px",width:len+"px",transform:`rotate(${Math.atan2(dy,dx)}rad)`});box.appendChild(bond);
    const atom=document.createElement("i");atom.className="atom";
    Object.assign(atom.style,{left:(p[0]-5)+"px",top:(p[1]-5)+"px"});
    if(i===2&&smiles.includes("N"))atom.style.background="#3568b0";
    if(i===5&&smiles.includes("O"))atom.style.background="#bd312a";
    box.appendChild(atom);
  });
}
function renderScores(round){
  el("scores").innerHTML=keys.map(k=>{
    const v=round.winner.scores[k],old=round.previous_scores?.[k]||0;
    return `<div class="score"><span>${k}</span><div class="track"><i class="previous" style="width:${old*100}%"></i><i class="current" style="width:${v*100}%"></i></div><b>${v.toFixed(2)}</b></div>`;
  }).join("");
}
function confidenceBadge(item){
  if(typeof item.confidence!=="number")return"";
  const basis=(item.confidence_basis||"Score separation within this candidate batch.").replaceAll('"',"&quot;");
  return`<span class="confidence" title="${basis}">${Math.round(item.confidence*100)}% confidence</span>`;
}
function renderTranscript(round){
  el("transcript").innerHTML=round.transcript.map(m=>{
    const veto=m.verdict==="veto"?" veto":m.agent==="orchestrator"?" orchestrator":"";
    return `<div class="message${veto}"><b>${m.agent}${m.verdict?` · ${m.verdict}`:""} ${confidenceBadge(m)}</b>${m.reason||m.rationale}</div>`;
  }).join("");
}
function renderChanges(round){
  el("parentSmiles").textContent=round.parent;
  const notes=round.changes?.structural||["Selected a new analogue."];
  const outcomes=round.changes?.outcome||[];
  el("changeNotes").innerHTML=`<ul class="change-note">${notes.map(x=>`<li>${x}</li>`).join("")}</ul>${outcomes.map(x=>`<div class="outcome">${x}</div>`).join("")}`;
}
function renderAssays(round){
  const assays=round.recommended_assays||[];
  el("assayList").innerHTML=assays.map(item=>`<article class="assay-item"><div class="assay-priority">${item.priority}</div><div><h3>${item.assay}</h3><p>${item.purpose}</p><p class="assay-readout"><b>Readout:</b> ${item.readout}</p></div></article>`).join("");
}
function renderCandidateReview(round){
  const reviews=round.agent_candidate_review||[];
  el("agentCandidateReview").innerHTML=reviews.map(review=>`<section class="agent-column"><h3><span>${review.agent}</span>${confidenceBadge(review)}</h3>${review.candidates.map(candidate=>`<details class="candidate-row"><summary><strong>${candidate.display_name}</strong><small>${candidate.summary}</small><span class="mini ${candidate.disposition}">${candidate.disposition}</span></summary><div class="candidate-explanation"><b>Change</b><p>${candidate.change}</p><b>Decision</b><p>${candidate.reason}</p></div></details>`).join("")}</section>`).join("");
}
function render(i){
  current=i;const round=run.rounds[i];
  el("roundLabel").textContent=`ROUND ${round.round} / ${run.rounds.length}`;
  el("candidateName").textContent=round.winner.display_name||"Selected haloperidol analogue";
  el("candidateId").textContent=`Selected winner · internal ID ${round.winner.id}`;
  el("smiles").textContent=round.winner.smiles;
  el("status").textContent=round.round<run.rounds.length?`This analogue becomes the parent for round ${round.round+1}`:"Optimization replay complete";
  el("previous").disabled=i===0;
  el("next").disabled=i===run.rounds.length-1&&unlocked===run.rounds.length-1;
  el("next").textContent=i<unlocked?"Next →":unlocked<run.rounds.length-1?`Complete round ${unlocked+2} →`:"Optimization complete";
  document.querySelectorAll(".round-button").forEach((button,index)=>button.classList.toggle("active",index===i));
  moleculeSketch(round.winner.smiles);set3D(round.winner.smiles,round.parent,round.structure_3d);
  el("viewerNote").textContent=round.structure_3d
    ?`Amber outlines changed atoms or deletion attachment sites. ${round.structure_3d.formula} · ${round.structure_3d.geometry} · explicit hydrogens.`
    :"Amber marks estimated molecular changes. RDKit geometry was unavailable, so this is a schematic fallback.";
  renderScores(round);renderChanges(round);renderAssays(round);renderTranscript(round);renderCandidateReview(round);
}
function showSeed(reset=true){
  current=-1;if(reset)unlocked=-1;el("roundLabel").textContent="STARTING MOLECULE";
  el("candidateName").textContent=run.seed_name;el("candidateId").textContent="Unmodified starting drug";
  el("smiles").textContent=run.seed;el("status").textContent="Complete round 1 to evaluate the first candidate batch";
  moleculeSketch(run.seed);set3D(run.seed,run.seed,run.seed_structure_3d);
  el("viewerNote").textContent=run.seed_structure_3d
    ?`Starting structure · ${run.seed_structure_3d.formula} · ${run.seed_structure_3d.geometry} · explicit hydrogens.`
    :"Starting structure. RDKit geometry was unavailable, so this is a schematic fallback.";
  renderScores({winner:{scores:run.seed_scores},previous_scores:null});
  el("changeNotes").innerHTML=`<div class="outcome">No changes yet. The optimization begins from ${run.seed_name}.</div>`;
  el("assayList").innerHTML=`<article class="assay-item"><div class="assay-priority">Baseline</div><div><h3>Reference compound characterization</h3><p>Confirm ${run.seed_name} identity, purity, and baseline assay performance before comparing analogues.</p><p class="assay-readout"><b>Readout:</b> LC–MS identity and reference ${run.target}/hERG controls.</p></div></article>`;
  el("agentCandidateReview").innerHTML='<div class="outcome">No candidates have been reviewed. Complete round 1 to reveal each agent’s candidate list, preferences, and safety vetoes.</div>';
  el("parentSmiles").textContent=run.seed;el("transcript").innerHTML='<div class="message"><b>READY</b>Complete round 1 to ask the three specialists and orchestrator to evaluate six analogues.</div>';
  el("previous").disabled=true;el("next").disabled=false;
  el("next").textContent=unlocked>=0?"Next →":"Complete round 1 →";
  document.querySelectorAll(".round-button").forEach((button,index)=>{button.disabled=index>unlocked;button.classList.remove("active")});
}
function advance(){
  if(current<unlocked){render(current+1);return}
  if(unlocked<run.rounds.length-1){unlocked++;document.querySelector(`.round-button[data-index="${unlocked}"]`).disabled=false;render(unlocked)}
}
function showDrug(drug){
  selectedDrug=drug;el("drugName").textContent=drug.name;el("drugTarget").textContent=drug.target;
  el("drugSmiles").textContent=drug.smiles;el("drugStructure").src=drug.structure_url;
  el("drugStructure").onerror=()=>{el("drugStructure").alt=`Structure image unavailable; SMILES shown for ${drug.name}`};
  el("searchMessage").textContent="";el("seedName").textContent=drug.name;
}
function installRun(data){
  run=data;unlocked=-1;current=-1;
  el("seedName").textContent=run.seed_name;el("councilTitle").textContent=`${run.target} Safety Council`;
  el("missionTarget").textContent=run.target;el("targetRationale").textContent=run.target_rationale;
  el("optimizationGoal").textContent=run.optimization_goal;el("hardConstraint").textContent=run.hard_constraint;
  el("notice").textContent=run.oracle_notice;
  el("roundButtons").innerHTML=run.rounds.map((round,index)=>`<button class="round-button" data-index="${index}" disabled title="View completed round ${round.round}">${round.round}</button>`).join("");
  showSeed();
}
function searchDrug(query){
  const q=query.trim().toLowerCase();
  const found=drugs.find(d=>d.name.toLowerCase().includes(q)||(d.aliases||[]).some(a=>a.toLowerCase().includes(q)));
  if(found)showDrug(found);else el("searchMessage").textContent=`No local match for “${query}”. Add the compound to fixtures/drugs.json.`;
}
Promise.all([fetch("../static_runs/haloperidol.json").then(r=>r.json()),fetch("../static_runs/drugs.json").then(r=>r.json())]).then(([data,catalog])=>{
  drugs=catalog;showDrug(drugs[0]);
  el("roundButtons").onclick=event=>{const button=event.target.closest(".round-button");if(button&&!button.disabled)render(Number(button.dataset.index))};
  el("previous").onclick=()=>{if(current>0)render(current-1);else showSeed(false)};
  el("next").onclick=advance;
  el("drugSearch").onsubmit=event=>{event.preventDefault();searchDrug(el("drugQuery").value)};
  el("runDrug").onclick=()=>{const file=selectedDrug?.run_file||"haloperidol.json";fetch(`../static_runs/${file}`).then(r=>r.json()).then(installRun).catch(err=>el("searchMessage").textContent=`Could not load ${selectedDrug.name}: ${err}`)};
  el("replay").onclick=()=>showSeed();setup3DControls();installRun(data);
}).catch(err=>el("notice").textContent=`Could not load replay: ${err}`);
