"""Headless figures: the Pareto plot, the score bars, the molecule image.

Importable with no Streamlit and no display, so this doubles as the fifth item
on the cut list -- if the UI has to go, `python -m frontend.plots demo_run.json`
still produces the two pictures the demo actually needs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot: no display in a container or on a demo laptop
import matplotlib.pyplot as plt  # noqa: E402

from core.loop import pareto_front  # noqa: E402

AXIS_LABELS = {
    "affinity": "target retention",
    "solubility": "solubility",
    "bbb": "BBB penetration",
    "herg": "hERG safety",
    "sa": "synthesisability",
}

PROFILE_AXES = ("affinity", "solubility", "herg", "sa")

VETO_COLOUR = "#c1121f"
FINAL_COLOUR = "#1d3557"
EARLY_COLOUR = "#adb5bd"


def axis_labels_for_run(run: dict) -> dict:
    """Labels that reflect the selected target and exposure objective."""
    labels = dict(AXIS_LABELS)
    program = run.get("program") or {}
    target = program.get("target") or run.get("target") or "target"
    labels["affinity"] = f"{target} retention"
    labels["bbb"] = {
        "avoid": "BBB avoidance",
        "neutral": "BBB neutral",
    }.get(program.get("bbb_goal"), "BBB penetration")
    return labels


def pareto_figure(run: dict, axes: tuple[str, str] | None = None, figsize=(6.4, 5.2)):
    """Round 1 in grey, the final round in colour, vetoed molecules marked.

    The vetoed points are the argument for the architecture being visible at a
    glance: several of them sit outside the final front, which is to say the
    optimiser wanted them and safety refused.
    """
    axes = tuple(axes or run.get("pareto_axes") or ("solubility", "herg"))
    x_axis, y_axis = axes
    labels = axis_labels_for_run(run)
    history = run["history"]
    last_round = max(h["round"] for h in history)

    early = [h for h in history if h["round"] <= 1 and h["origin"] != "vetoed"]
    late = [h for h in history if h["origin"] != "vetoed"]
    vetoed = [h for h in history if h["origin"] == "vetoed"]

    figure, axis = plt.subplots(figsize=figsize)

    def scatter(points, **kwargs):
        if points:
            axis.scatter(
                [p["normalised"][x_axis] for p in points],
                [p["normalised"][y_axis] for p in points],
                **kwargs,
            )

    def draw_front(points, colour, label, linestyle="-"):
        front = pareto_front(points, axes)
        if len(front) >= 2:
            axis.plot(
                [p["normalised"][x_axis] for p in front],
                [p["normalised"][y_axis] for p in front],
                linestyle,
                color=colour,
                linewidth=2.0,
                label=label,
                zorder=2,
            )
        elif front:
            # A one-point front is the normal case in round 1: draw it as a
            # marker, not a line, and make it distinct from the point cloud.
            axis.scatter(
                [front[0]["normalised"][x_axis]],
                [front[0]["normalised"][y_axis]],
                facecolors=colour,
                edgecolors=colour,
                marker="s",
                s=110,
                label=label,
                zorder=2,
            )

    scatter(late, color=EARLY_COLOUR, s=26, alpha=0.7, zorder=1,
            label="all molecules evaluated")
    draw_front(early, "#6c757d", "round 1 front", linestyle="--")
    draw_front(late, FINAL_COLOUR, f"round {last_round} front")
    scatter(vetoed, color=VETO_COLOUR, s=70, marker="x", linewidths=2.0,
            label="vetoed on hERG", zorder=3)

    seed = next((h for h in history if h["round"] == 0), None)
    if seed:
        axis.scatter(
            [seed["normalised"][x_axis]], [seed["normalised"][y_axis]],
            facecolors="none", edgecolors="black", s=160, linewidths=1.6,
            label="seed", zorder=4,
        )

    axis.set_xlabel(labels.get(x_axis, x_axis) + "  (normalised, higher better)")
    axis.set_ylabel(labels.get(y_axis, y_axis) + "  (normalised, higher better)")
    axis.set_title(
        f"Pareto front, rounds 1-{last_round}"
        + ("   [SURROGATE ORACLES]" if run["provenance"]["any_surrogate"] else "")
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(loc="lower left", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    return figure


def score_bars_figure(current: dict, previous: dict | None = None,
                      figsize=(6.4, 3.2), labels: dict | None = None):
    """Four interpretable bars, with the previous round ghosted in grey."""
    figure, axis = plt.subplots(figsize=figsize)
    positions = range(len(PROFILE_AXES))
    axis_labels = labels or AXIS_LABELS
    tick_labels = [axis_labels.get(a, a) for a in PROFILE_AXES]

    if previous:
        axis.bar(
            positions, [previous["normalised"][a] for a in PROFILE_AXES],
            color=EARLY_COLOUR, alpha=0.55, width=0.72, label="previous round",
        )
    axis.bar(
        positions, [current["normalised"][a] for a in PROFILE_AXES],
        color=FINAL_COLOUR, width=0.44, label="current",
    )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(tick_labels, fontsize=8, rotation=15)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("normalised (higher better)")
    axis.grid(axis="y", alpha=0.25, linestyle=":")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def molecule_png(smiles: str, path: Path, size=(420, 320)) -> Path:
    """RDKit depiction to a PNG file."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    Draw.MolToImage(mol, size=size).save(str(path))
    return path


def main() -> None:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_run.json")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads(source.read_text())

    pareto_figure(run).savefig(out_dir / "pareto.png", dpi=160)
    rounds = run["round_records"]
    previous = rounds[0]["parent"] if rounds else None
    score_bars_figure(
        run["final"], previous, labels=axis_labels_for_run(run)
    ).savefig(out_dir / "scores.png", dpi=160)
    molecule_png(run["seed"], out_dir / "seed.png")
    molecule_png(run["final"]["smiles"], out_dir / "final.png")

    print(f"wrote {out_dir}/pareto.png scores.png seed.png final.png")
    if run["provenance"]["any_surrogate"]:
        print("NOTE: figures were produced from surrogate oracles.")


if __name__ == "__main__":
    main()
