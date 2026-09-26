"""Scientist-facing Streamlit demo for manual round-by-round review."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core.contract import AXES
from core.drugs import HALOPERIDOL, search
from core.loop import run
from frontend.molecule3d import viewer_html
from frontend.plots import AXIS_LABELS, molecule_png, pareto_figure, score_bars_figure
from frontend.science import (
    candidate_reviews,
    chosen_candidate,
    describe_change,
    recommended_assays,
)

DEFAULT_RUN = Path("demo_run.json")

VERDICT_STYLE = {
    "support": ("#2d6a4f", "", ""),
    "object": ("#c1121f", "", ""),
    "veto": ("#7f0000", "**", "**"),
}

CSS = """
<style>
.mission-grid{display:grid;grid-template-columns:.55fr 1.2fr 1.2fr 1fr;
 border:1px solid #e1ded4;border-radius:10px;overflow:hidden;margin:.6rem 0 1rem}
.mission-cell{padding:14px 16px;border-left:1px solid #e1ded4}
.mission-cell:first-child{border-left:0}.mission-cell.constraint{background:#fff0ee}
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
@media(max-width:900px){.mission-grid{grid-template-columns:1fr 1fr}}
</style>
"""


def load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def molecule_image(smiles: str):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        return str(molecule_png(smiles, Path(handle.name)))


def provenance_banner(run_record: dict) -> None:
    provenance = run_record["provenance"]
    if provenance["any_surrogate"]:
        surrogate_axes = [
            axis
            for axis in AXES
            if "surrogate" in str(provenance.get(axis, "")).lower()
        ]
        st.error(
            "**Surrogate oracle active.** "
            + ", ".join(surrogate_axes)
            + " is not a validated activity prediction. ADMET and SA provenance "
            "are shown separately; do not attach published accuracy claims until "
            "the model verification gate passes."
        )
    else:
        st.success(
            f"Oracles: affinity `{provenance['affinity']}`, "
            f"ADMET `{provenance['solubility']}`, SA `{provenance['sa']}`."
        )


def mission_panel(drug) -> None:
    st.markdown(
        f"""
<div class="mission-grid">
 <div class="mission-cell"><span class="mission-label">TARGET</span>
  <p><b>{html.escape(drug.target)}</b></p></div>
 <div class="mission-cell"><span class="mission-label">WHY THIS TARGET</span>
  <p>{html.escape(drug.rationale)}</p></div>
 <div class="mission-cell"><span class="mission-label">OPTIMIZATION GOAL</span>
  <p>{html.escape(drug.objective)}</p></div>
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


def render_candidate_review(record: dict) -> None:
    st.subheader("Candidate review by agent")
    st.caption(
        "Expand a candidate to see its structural edit and why that agent "
        "preferred, rejected, vetoed, or declined to select it."
    )
    columns = st.columns(4)
    for column, review in zip(columns, candidate_reviews(record)):
        chunks = [f"<div class='agent-card'><div class='agent-title'>{review['agent']}</div>"]
        for candidate in review["candidates"]:
            disposition = candidate["disposition"].lower()
            chunks.append(
                "<details class='candidate'><summary>"
                f"<span class='candidate-name'>{html.escape(candidate['label'])}</span>"
                f"<span class='candidate-score'>{html.escape(candidate['score'])}</span>"
                f"<span class='badge {disposition}'>{html.escape(candidate['disposition'])}</span>"
                "</summary><div class='candidate-body'><b>CHANGE</b>"
                f"<p>{html.escape(candidate['change'])}</p><b>DECISION</b>"
                f"<p>{html.escape(candidate['reason'])}</p></div></details>"
            )
        chunks.append("</div>")
        column.markdown("".join(chunks), unsafe_allow_html=True)


def render_assays(record: dict) -> None:
    st.subheader("Suggested wet-lab verification")
    st.caption("Experiments ranked from this round's predicted changes.")
    for assay in recommended_assays(record):
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

    with st.sidebar:
        st.header("Starting compound")
        query = st.text_input("Search generic or brand name", "Haloperidol")
        matches = search(query)
        if not matches:
            st.warning("No compound found.")
            st.stop()
        drug = st.selectbox("Compound", matches, format_func=lambda item: item.name)
        st.image(molecule_image(drug.smiles), caption=f"{drug.name} · {drug.target}")
        st.code(drug.smiles, language=None)

        st.header("Run")
        path = Path(st.text_input("Cached run", str(DEFAULT_RUN)))
        rounds = st.slider("Rounds for a new live run", 1, 8, 5)
        live = st.button("Generate live run", type="primary", disabled=not drug.optimizable)
        if st.button("Reset round review"):
            st.session_state["completed_rounds"] = 0
            st.session_state["selected_round"] = 0

    st.title(f"{drug.name} · {drug.target} lead optimization")
    mission_panel(drug)

    if not drug.optimizable:
        st.warning(drug.limitation)
        left, right = st.columns(2)
        with left:
            st.subheader("2D structure")
            st.image(molecule_image(drug.smiles))
        with right:
            st.subheader("Interactive 3D structure")
            components.html(viewer_html(drug.smiles), height=390)
        st.stop()

    if live:
        with st.spinner("Scoring candidates and running five-agent decisions..."):
            run_record = run(seed=drug.smiles, rounds=rounds, verbose=False)
        path.write_text(json.dumps(run_record, indent=2))
        st.session_state["run"] = run_record
        st.session_state["run_key"] = f"{path.resolve()}:{drug.name}"
        st.session_state["completed_rounds"] = 0
        st.session_state["selected_round"] = 0

    run_key = f"{path.resolve()}:{drug.name}"
    if st.session_state.get("run_key") != run_key:
        loaded = load_run(path)
        if loaded is None:
            st.warning(f"No cached run at `{path}`. Generate one from the sidebar.")
            st.stop()
        st.session_state["run"] = loaded
        st.session_state["run_key"] = run_key
        st.session_state["completed_rounds"] = 0
        st.session_state["selected_round"] = 0

    run_record = st.session_state["run"]
    provenance_banner(run_record)

    if st.session_state["completed_rounds"] == 0:
        st.info("Starting at unmodified Haloperidol. Complete round 1 when you are ready.")
        left, right = st.columns(2)
        seed = run_record["history"][0]
        with left:
            st.subheader("Starting structure")
            st.image(molecule_image(seed["smiles"]))
            st.pyplot(score_bars_figure(seed))
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
            AXIS_LABELS[axis],
            f"{chosen['normalised'][axis]:.2f}",
            f"{chosen['normalised'][axis] - seed_norm[axis]:+.2f} vs seed",
        )

    structure, viewer = st.columns(2)
    with structure:
        st.image(molecule_image(chosen["smiles"]))
        st.code(chosen["smiles"], language=None)
        st.markdown("**What changed**")
        st.write(describe_change(record, chosen))
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
        st.pyplot(score_bars_figure(chosen, record["parent"]))
    with transcript:
        st.subheader("Agent deliberation")
        render_transcript(record)

    render_candidate_review(record)
    render_assays(record)

    st.subheader("Pareto front through completed rounds")
    st.pyplot(pareto_figure(partial_run(run_record, st.session_state["completed_rounds"])))


if __name__ == "__main__":
    main()
