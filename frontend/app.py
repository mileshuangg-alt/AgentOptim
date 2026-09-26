"""Scientist-facing Streamlit demo for manual round-by-round review."""

from __future__ import annotations

import html
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contract import AXES, canonical
from core.drugs import DRUGS, HALOPERIDOL, Drug, from_pubchem
from core.loop import run
from core.pubchem import PubChemError, fetch_compound, search_names
from frontend.molecule3d import viewer_html
from frontend.plots import (
    axis_labels_for_run,
    molecule_png,
    pareto_figure,
    score_bars_figure,
)
from frontend.science import (
    candidate_reviews,
    chosen_candidate,
    describe_change,
    recommended_assays,
)

DEFAULT_RUN = ROOT / "demo_run.json"
RUNS_DIR = DEFAULT_RUN.parent / "runs"

VERDICT_STYLE = {
    "support": ("#2d6a4f", "", ""),
    "object": ("#c1121f", "", ""),
    "veto": ("#7f0000", "**", "**"),
}

CSS = """
<style>
.mission-grid{display:grid;grid-template-columns:repeat(3,1fr);
 border:1px solid #e1ded4;border-radius:10px;overflow:hidden;margin:.6rem 0 1rem}
.mission-cell{padding:14px 16px;border-left:1px solid #e1ded4;
 border-top:1px solid #e1ded4}
.mission-cell:nth-child(-n+3){border-top:0}.mission-cell:nth-child(3n+1){border-left:0}
.mission-cell.constraint{background:#fff0ee}
.mission-label{font-size:.65rem;font-weight:800;letter-spacing:.12em;color:#287255}
.mission-cell p{font-size:.82rem;margin:.25rem 0 0;color:#48564f}
.mission-cell.constraint p,.mission-cell.constraint .mission-label{color:#9b332e}
.agent-card{border:1px solid #e1ded4;border-radius:9px;overflow:hidden}
.agent-title{padding:9px 11px;background:#eff3ef;font-weight:750}
details.candidate{border-top:1px solid #ebe8df}
details.candidate summary{position:relative;list-style:none;cursor:pointer;
 padding:9px 40px 9px 11px}
details.candidate summary::-webkit-details-marker{display:none}
details.candidate summary:before{content:"";position:absolute;right:11px;top:50%;
 width:21px;height:21px;border:1px solid #cfd5d0;border-radius:50%;
 transform:translateY(-50%);background:white}
details.candidate summary:after{content:"";position:absolute;right:19px;top:50%;
 width:6px;height:6px;border-right:2px solid #287255;border-bottom:2px solid #287255;
 transform:translateY(-65%) rotate(45deg)}
details.candidate[open] summary:before{background:#287255;border-color:#287255}
details.candidate[open] summary:after{border-color:white;transform:translateY(-35%) rotate(225deg)}
details.candidate summary:hover{background:#f7f8f5}
.candidate-name{display:block;font-size:.76rem;font-weight:700}
.candidate-score{display:block;font-size:.7rem;color:#727b76}
.badge{display:inline-block;margin-top:4px;padding:2px 7px;border-radius:20px;
 font-size:.57rem;font-weight:800;text-transform:uppercase;background:#ebeae5;color:#626964}
.badge.vetoed,.badge.ineligible{background:#ffe3df;color:#a8332e}
.badge.preferred,.badge.selected{background:#dcefe5;color:#287255}
.candidate-body{padding:8px 12px 12px;background:#faf9f5;border-top:1px solid #ebe8df}
.candidate-body b{font-size:.6rem;letter-spacing:.08em;color:#287255}
.candidate-body p{font-size:.72rem;margin:2px 0 8px;color:#4e5d55}
@media(max-width:900px){.mission-grid{grid-template-columns:1fr 1fr}
 .mission-cell:nth-child(3n+1){border-left:1px solid #e1ded4}
 .mission-cell:nth-child(2n+1){border-left:0}}
</style>
"""


def load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pubchem_names(query: str) -> list[str]:
    return search_names(query)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_pubchem_compound(name: str):
    return fetch_compound(name)


def activate_drug(drug: Drug) -> None:
    """Switch structures and reset target-context widgets together."""
    st.session_state["active_drug"] = drug
    st.session_state["program_target"] = drug.target
    st.session_state["program_rationale"] = drug.rationale
    st.session_state["program_objective"] = drug.objective
    st.session_state["program_bbb_goal"] = drug.bbb_goal
    if canonical(drug.smiles) == canonical(HALOPERIDOL.smiles):
        cache_path = DEFAULT_RUN
    else:
        slug = "".join(
            character.lower() if character.isalnum() else "-"
            for character in drug.name
        ).strip("-")
        cache_path = RUNS_DIR / f"{slug or 'compound'}_run.json"
    st.session_state["cached_run_path"] = str(cache_path)
    st.session_state.pop("run", None)
    st.session_state.pop("run_key", None)
    st.session_state["completed_rounds"] = 0
    st.session_state["selected_round"] = 0


def activity_strategy_preview(target: str) -> str:
    key = "".join(character for character in target.lower() if character.isalnum())
    if key in {"drd2", "d2", "dopamined2", "dopamined2receptor"}:
        return (
            "Use the PyTDC DRD2 oracle when available; otherwise use similarity "
            "to known DRD2 ligands and label it as a proxy."
        )
    return (
        f"No validated {target or 'target-specific'} activity oracle is connected. "
        "Use similarity to this starting structure only to limit scaffold drift; "
        "wet-lab target engagement is required."
    )


def run_matches_program(run_record: dict, drug: Drug, program: dict) -> bool:
    try:
        same_seed = canonical(run_record.get("seed", "")) == canonical(drug.smiles)
    except ValueError:
        return False
    saved_program = run_record.get("program") or {}
    same_target = (saved_program.get("target") or run_record.get("target")) == program["target"]
    same_bbb_goal = saved_program.get("bbb_goal", "penetrate") == program["bbb_goal"]
    same_justification = (
        not saved_program
        or (
            saved_program.get("rationale") == program["rationale"]
            and saved_program.get("objective") == program["objective"]
        )
    )
    return same_seed and same_target and same_bbb_goal and same_justification


def molecule_image(smiles: str):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        return str(molecule_png(smiles, Path(handle.name)))


def provenance_banner(run_record: dict) -> None:
    provenance = run_record["provenance"]
    if "similarity" in str(provenance.get("affinity", "")).lower():
        st.warning(
            "**Target activity uses a structural proxy.** "
            + provenance.get(
                "activity_note",
                "Similarity does not establish binding or functional activity.",
            )
        )
    if any(provenance.get(axis) == "surrogate" for axis in ("solubility", "bbb", "herg")):
        st.error("A labelled physicochemical surrogate is active for an ADMET axis.")
    else:
        st.success(
            f"ADMET `{provenance['solubility']}` · synthetic accessibility "
            f"`{provenance['sa']}`."
        )


def mission_panel(drug: Drug, program: dict) -> None:
    exposure = {
        "penetrate": "Favor CNS penetration.",
        "avoid": "Favor peripheral restriction and lower CNS exposure.",
        "neutral": "Do not use BBB penetration to rank candidates.",
    }[program["bbb_goal"]]
    st.markdown(
        f"""
<div class="mission-grid">
 <div class="mission-cell"><span class="mission-label">TARGET</span>
  <p><b>{html.escape(program["target"])}</b></p></div>
 <div class="mission-cell"><span class="mission-label">WHY THIS TARGET</span>
  <p>{html.escape(program["rationale"])}</p></div>
 <div class="mission-cell"><span class="mission-label">OPTIMIZATION GOAL</span>
  <p>{html.escape(program["objective"])}</p></div>
 <div class="mission-cell"><span class="mission-label">EXPOSURE GOAL</span>
  <p>{html.escape(exposure)}</p></div>
 <div class="mission-cell"><span class="mission-label">ACTIVITY EVIDENCE</span>
  <p>{html.escape(program["activity_strategy"])}</p></div>
 <div class="mission-cell constraint"><span class="mission-label">HARD CONSTRAINT</span>
  <p>Safety may veto candidates outright; the orchestrator cannot restore them.</p></div>
</div>""",
        unsafe_allow_html=True,
    )


def render_transcript(record: dict) -> None:
    for review in record["reviews"]:
        colour, open_mark, close_mark = VERDICT_STYLE.get(
            review["verdict"], ("#333333", "", "")
        )
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding:.4rem .8rem;"
            f"margin-bottom:.5rem;background:rgba(0,0,0,.02)'>"
            f"<span style='color:{colour};font-weight:600'>"
            f"{html.escape(review['agent'])} · {review['verdict'].upper()}</span><br>"
            f"<span style='color:{colour}'>{open_mark}"
            f"{html.escape(review['reason'])}{close_mark}</span></div>",
            unsafe_allow_html=True,
        )
    decision = record["decision"]
    st.markdown(
        "<div style='border-left:4px solid #1d3557;padding:.4rem .8rem;"
        "background:rgba(29,53,87,.06)'><b style='color:#1d3557'>"
        f"Orchestrator</b><br>{html.escape(decision['rationale'])}</div>",
        unsafe_allow_html=True,
    )
    if decision.get("forced_second_best"):
        st.warning("Safety vetoed the leading candidate; the orchestrator selected the next viable option.")


def render_candidate_review(record: dict, program: dict) -> None:
    st.subheader("Candidate review by agent")
    st.caption(
        "Expand a candidate to see its structural edit and why that agent "
        "preferred, rejected, vetoed, or declined to select it."
    )
    columns = st.columns(4)
    for column, review in zip(columns, candidate_reviews(record, program)):
        chunks = [f"<div class='agent-card'><div class='agent-title'>{review['agent']}</div>"]
        for candidate in review["candidates"]:
            disposition = candidate["disposition"].lower()
            chunks.append(
                "<details class='candidate'><summary>"
                f"<span class='candidate-name'>{html.escape(candidate['label'])}</span>"
                f"<span class='candidate-score'>{html.escape(candidate['score'])}</span>"
                f"<span class='badge {disposition}'>{html.escape(candidate['disposition'])}</span>"
                "</summary><div class='candidate-body'><b>CHANGE</b>"
                f"<p>{html.escape(candidate['change'])}</p><b>TARGET CONTEXT</b>"
                f"<p>{html.escape(candidate['target_context'])}</p><b>DECISION</b>"
                f"<p>{html.escape(candidate['reason'])}</p></div></details>"
            )
        chunks.append("</div>")
        column.markdown("".join(chunks), unsafe_allow_html=True)


def render_assays(record: dict, program: dict) -> None:
    st.subheader("Suggested wet-lab verification")
    st.caption("Experiments ranked from this round's predicted changes.")
    for assay in recommended_assays(record, program):
        with st.container(border=True):
            st.markdown(f"**{assay['name']}** · `{assay['priority']}`")
            st.write(assay["why"])
            st.caption(f"Readout: {assay['readout']}")


def partial_run(run_record: dict, completed: int) -> dict:
    partial = dict(run_record)
    partial["round_records"] = run_record["round_records"][:completed]
    partial["history"] = [
        entry for entry in run_record["history"] if entry["round"] <= completed
    ]
    return partial


def select_round(run_record: dict) -> int:
    completed = st.session_state["completed_rounds"]
    selected = st.session_state["selected_round"]
    left, middle, right = st.columns([1, 4, 1])
    with left:
        if st.button("← Previous", disabled=selected <= 1, use_container_width=True):
            st.session_state["selected_round"] -= 1
            st.rerun()
    with middle:
        buttons = st.columns(run_record["rounds"])
        for index, button_column in enumerate(buttons, 1):
            with button_column:
                if st.button(
                    str(index),
                    key=f"round_{index}",
                    disabled=index > completed,
                    type="primary" if index == selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_round"] = index
                    st.rerun()
    with right:
        if completed < run_record["rounds"] and selected == completed:
            label = f"Complete {completed + 1} →"
            if st.button(label, type="primary", use_container_width=True):
                st.session_state["completed_rounds"] += 1
                st.session_state["selected_round"] = completed + 1
                st.rerun()
        elif st.button(
            "Next →",
            disabled=selected >= completed,
            use_container_width=True,
        ):
            st.session_state["selected_round"] += 1
            st.rerun()
    return st.session_state["selected_round"]


def main() -> None:
    st.set_page_config(page_title="AgentOptim lead optimization", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if "active_drug" not in st.session_state:
        activate_drug(HALOPERIDOL)
    st.session_state.setdefault("completed_rounds", 0)
    st.session_state.setdefault("selected_round", 0)

    with st.sidebar:
        st.header("Find a starting compound")
        query = st.text_input(
            "Search PubChem by drug or compound name",
            placeholder="e.g. aspirin, metformin, caffeine",
        )
        if st.button("Search PubChem", use_container_width=True):
            try:
                names = cached_pubchem_names(query)
                st.session_state["pubchem_matches"] = list(
                    dict.fromkeys(([query.strip()] if query.strip() else []) + names)
                )
            except PubChemError as exc:
                st.session_state["pubchem_matches"] = []
                st.error(str(exc))

        pubchem_matches = st.session_state.get("pubchem_matches", [])
        if pubchem_matches:
            selected_name = st.selectbox("PubChem matches", pubchem_matches)
            if st.button("Load PubChem structure", use_container_width=True):
                try:
                    activate_drug(from_pubchem(cached_pubchem_compound(selected_name)))
                    st.rerun()
                except PubChemError as exc:
                    st.error(str(exc))

        with st.expander("Offline examples"):
            example = st.selectbox(
                "Curated compound", DRUGS, format_func=lambda item: item.name
            )
            if st.button("Use example", use_container_width=True):
                activate_drug(example)
                st.rerun()

        drug = st.session_state["active_drug"]
        st.divider()
        st.image(molecule_image(drug.smiles), caption=f"{drug.name} · {drug.source}")
        st.code(drug.smiles, language=None)
        if drug.original_smiles and drug.original_smiles != drug.smiles:
            st.caption(
                "PubChem returned a salt or mixture. The displayed largest "
                "fragment is the structure used for scoring and optimization."
            )
            with st.expander("Original PubChem SMILES"):
                st.code(drug.original_smiles, language=None)
        if drug.pubchem_cid:
            st.markdown(
                f"[Open PubChem record](https://pubchem.ncbi.nlm.nih.gov/compound/"
                f"{drug.pubchem_cid})"
            )

        st.header("Target context")
        target = st.text_input(
            "Target or phenotype",
            key="program_target",
            placeholder="e.g. COX-2, EGFR, bacterial growth",
        ).strip()
        rationale = st.text_area(
            "Why this target matters",
            key="program_rationale",
            placeholder=(
                "State the disease or biological rationale. This is shown with "
                "every optimization decision."
            ),
        ).strip()
        objective = st.text_area(
            "Optimization objective",
            key="program_objective",
        ).strip()
        bbb_goal = st.selectbox(
            "CNS exposure goal",
            options=("penetrate", "avoid", "neutral"),
            format_func={
                "penetrate": "Favor BBB penetration",
                "avoid": "Favor peripheral restriction",
                "neutral": "Do not optimize BBB exposure",
            }.get,
            key="program_bbb_goal",
        )
        program = {
            "compound": drug.name,
            "source": drug.source,
            "target": target,
            "rationale": rationale,
            "objective": objective,
            "bbb_goal": bbb_goal,
            "activity_strategy": activity_strategy_preview(target),
        }
        program_ready = bool(target and rationale and objective and drug.optimizable)
        if drug.limitation:
            st.info(drug.limitation)
        if not program_ready:
            st.caption(
                "Enter the target, its biological rationale, and the optimization "
                "objective before generating candidates."
            )

        st.header("Run")
        path = Path(st.text_input("Cached run", key="cached_run_path"))
        rounds = st.slider("Rounds for a new live run", 1, 8, 5)
        live = st.button(
            "Generate live run",
            type="primary",
            disabled=not program_ready,
            use_container_width=True,
        )
        if st.button("Reset round review"):
            st.session_state["completed_rounds"] = 0
            st.session_state["selected_round"] = 0

    target_label = target or "target context required"
    st.title(f"{drug.name} · {target_label} lead optimization")
    mission_panel(drug, program)

    if live:
        st.session_state.pop("run", None)
        st.session_state.pop("run_key", None)
        try:
            with st.spinner("Scoring candidates and running specialist decisions..."):
                run_record = run(
                    seed=drug.smiles,
                    rounds=rounds,
                    verbose=False,
                    target=target,
                    compound_name=drug.name,
                    rationale=rationale,
                    objective=objective,
                    bbb_goal=bbb_goal,
                    source=drug.source,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(run_record, indent=2))
            st.session_state["run"] = run_record
            st.session_state["completed_rounds"] = 0
            st.session_state["selected_round"] = 0
        except Exception as exc:
            st.error(f"Could not generate this run: {exc}")

    run_key = (
        f"{path.resolve()}:{canonical(drug.smiles)}:"
        f"{json.dumps(program, sort_keys=True)}"
    )
    if live and "run" in st.session_state:
        st.session_state["run_key"] = run_key

    if st.session_state.get("run_key") != run_key:
        loaded = load_run(path)
        if loaded is not None and run_matches_program(loaded, drug, program):
            st.session_state["run"] = loaded
            st.session_state["run_key"] = run_key
            st.session_state["completed_rounds"] = 0
            st.session_state["selected_round"] = 0
        else:
            st.session_state.pop("run", None)

    if "run" not in st.session_state:
        st.info(
            "Structure loaded. Complete the target context and generate a live "
            "run to begin round-by-round optimization."
        )
        left, right = st.columns(2)
        with left:
            st.subheader("2D structure")
            st.image(molecule_image(drug.smiles))
        with right:
            st.subheader("Interactive 3D structure")
            components.html(viewer_html(drug.smiles), height=390)
        return

    run_record = st.session_state["run"]
    run_program = run_record.get("program") or program
    display_run = dict(run_record, program=run_program)
    labels = axis_labels_for_run(display_run)
    provenance_banner(run_record)

    if st.session_state["completed_rounds"] == 0:
        st.info(
            f"Starting at unmodified {drug.name} for {run_program['target']}. "
            "Complete round 1 when you are ready."
        )
        left, right = st.columns(2)
        seed = run_record["history"][0]
        with left:
            st.subheader("Starting structure")
            st.image(molecule_image(seed["smiles"]))
            st.pyplot(score_bars_figure(seed, labels=labels))
        with right:
            st.subheader("Interactive 3D structure")
            components.html(viewer_html(seed["smiles"]), height=390)
        if st.button("Complete round 1 →", type="primary"):
            st.session_state["completed_rounds"] = 1
            st.session_state["selected_round"] = 1
            st.rerun()
        return

    selected = select_round(run_record)
    record = run_record["round_records"][selected - 1]
    chosen = chosen_candidate(record)
    seed_norm = run_record["history"][0]["normalised"]

    st.subheader(f"Round {selected} selected analogue")
    metric_columns = st.columns(5)
    for column, axis in zip(metric_columns, AXES):
        column.metric(
            labels[axis],
            f"{chosen['normalised'][axis]:.2f}",
            f"{chosen['normalised'][axis] - seed_norm[axis]:+.2f} vs seed",
        )

    structure, viewer = st.columns(2)
    with structure:
        st.image(molecule_image(chosen["smiles"]))
        st.code(chosen["smiles"], language=None)
        st.markdown("**What changed**")
        st.write(describe_change(record, chosen))
        st.markdown("**Why this is relevant to the target program**")
        st.write(
            f"For {run_program['target']}: {run_program['objective']} "
            f"{run_program['activity_strategy']}"
        )
    with viewer:
        st.markdown("**Interactive 3D structure**")
        components.html(
            viewer_html(chosen["smiles"], record["parent"]["smiles"]),
            height=390,
        )
        st.caption(
            "Amber marks changed atoms or the surviving attachment site of a "
            "removed atom. Drag to rotate; scroll to zoom."
        )

    profile, transcript = st.columns(2)
    with profile:
        st.subheader("ADMET and optimization profile")
        st.pyplot(score_bars_figure(chosen, record["parent"], labels=labels))
    with transcript:
        st.subheader("Agent deliberation")
        render_transcript(record)

    render_candidate_review(record, run_program)
    render_assays(record, run_program)

    st.subheader("Pareto front through completed rounds")
    st.pyplot(
        pareto_figure(
            partial_run(display_run, st.session_state["completed_rounds"])
        )
    )


if __name__ == "__main__":
    main()
