# AgentOptim — target-aware multi-agent lead optimisation

Specialist LLM agents optimising a molecule across competing properties, with
visible disagreement, a hard safety veto, and a Pareto front that moves.

The bundled run uses haloperidol against DRD2. The Streamlit app can search
PubChem for any compound and build a new target program around its structure.

```bash
pip install -r requirements.txt
python -m core.loop --rounds 5 --out demo_run.json   # run + cache
streamlit run frontend/app.py                        # manual round-by-round review
python -m scripts.serve_static                       # original static presentation
python -m frontend.plots demo_run.json figures/      # no-UI fallback
pytest -q
```

Optimization runs offline. PubChem search requires internet access, while the
curated examples and cached runs do not. Without an API key, the agents use
deterministic verdicts and the edit engine uses SMARTS mutations. Set
`ANTHROPIC_API_KEY` to get LLM-written argument and LLM-proposed chemistry.
AWS Bedrock is also supported through the same agent boundary:

```bash
export AGENT_LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID='<a Converse-compatible model enabled in your account>'
streamlit run frontend/app.py
```

The AWS CLI is optional. Boto3 uses its normal environment, profile, SSO, or
instance-role credential chain. Install `boto3` separately when using Bedrock.

The Streamlit interface begins at the named starting drug and unlocks each
round only when the user clicks **Complete round**. It includes:

- a prominent target, rationale, optimization goal, and hard-constraint panel;
- PubChem name search with the returned CID and canonical structure;
- required biological rationale and CNS exposure goal for each new program;
- RDKit 2D structures and a draggable, zoomable explicit-H ball-and-stick model
  selected from energy-minimized ETKDGv3 conformers;
- MCS-based highlighting of edited atoms and deletion attachment sites;
- ADMET changes, agent transcript, and a Pareto front through completed rounds;
- expandable candidate reviews with each edit, agent-specific rejection, and
  score-separation confidence;
- wet-lab assay suggestions tied to the predicted property changes.

Agent confidence reports how clearly the agent's scoring rule separates its
choice from the alternatives in that round. It is bounded at 95% and is not a
calibrated estimate of model accuracy or experimental success.

The original cream-and-green presentation is also retained as a dependency-free
static replay at `http://127.0.0.1:8000/static/index.html`. It uses the cached
Haloperidol and Ibuprofen runs in `static_runs/`, keeps the desktop card layout,
and switches to a compact grid on narrow screens. The round badge has its own
grid row so it cannot overlap the analogue title, while long SMILES wrap inside
the card.

For DRD2, the run uses PyTDC when available and otherwise labels the DRD2
similarity fallback. For every other target, it uses similarity to the selected
starting structure only as a scaffold-retention proxy. The UI repeats that
limitation in the mission panel, each candidate review, and wet-lab suggestions.

---

## Read this before demoing

Three findings from building this. Two of them contradict the plan it was
built from.

### 0. The real ADMET models are wired in; affinity is still a surrogate

`core/admet.py` now mirrors `app.py:featurize_one` from
[tuhinc5203/admet-property-prediction](https://github.com/tuhinc5203/admet-property-prediction)
exactly — largest-fragment salt stripping, 2048-bit Morgan r=2, and six
descriptors in the order **MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors,
NumRotatableBonds**. Note TPSA is third, not last; a reasonable guess at that
order gets the position wrong and the models return confident nonsense without
raising. `scripts/verify_models.py` can execute the source repo's three
featurization functions and compare their DataFrames directly.

The solubility, BBB, and hERG estimators are committed in `models/`. Their
manifest records the source commit and SHA-256 hashes, and the default
verification command checks 12 fixed predictions at a tolerance of `1e-6`.
Affinity still falls back to Tanimoto similarity against a panel of known DRD2
ligands when PyTDC's DRD2 oracle is unavailable.

### 1. An absolute hERG veto kills the demo in round 1

The real hERG model scores the seed at **P(blockade) = 0.865**. So does the
rest of the class: risperidone 0.923, chlorpromazine 0.912, aripiprazole 0.909.
The model is right — haloperidol carries a QT prolongation warning. But an
absolute 0.70 line vetoes the seed itself and then **0 of 14** first-round
analogues clear it. The loop has nothing to choose from, forever.

`HERG_VETO_MODE = "auto"` resolves the line per round against the incumbent:
absolute while the lead is clean, **relative** once it is already liable — no
analogue may be more cardiotoxic than the molecule you already have. That is
what a real programme does with a lead like this, and it has teeth: 7 of those
14 analogues are safer than the parent and 7 are worse, so the veto
discriminates within the series instead of blanketing it.

### 1b. The remaining surrogate

Every score record carries `provenance`, the CLI prints a warning, and the UI
shows a banner for any fallback. Verify the committed ADMET models before a
demo:

```bash
python -m scripts.verify_models
```

To audit the implementation against a local clone of the source:

```bash
python -m scripts.verify_models --source-repo /path/to/admet-property-prediction
```

The gate checks artifact hashes, the estimators' 2,054 stored feature names,
fixed predictions, and optional exact DataFrame equality with source
`app.py`. This matters because wrong descriptor order or a different Morgan
radius can return plausible numbers without raising an error.

### 2. The Pareto front is not plotted on affinity, and it cannot be

Haloperidol is a marketed antipsychotic. It sits at the **ceiling** of any DRD2
activity oracle, so the affinity axis has no headroom above the seed: every
edit can only lose potency, and a front seeded in the corner of that axis
cannot move outward. The first version of this plotted affinity × BBB and the
hypervolume was flat at 0.877 across five rounds.

The front is drawn on **solubility × hERG safety** — haloperidol's actual
liabilities, 0.24 and 0.40 normalised, both with room to improve. Affinity is
held as a **programme floor** (0.60) instead, reported next to every result, so
a front that moved by spending the molecule's potency is visible as the failure
it is rather than as a win.

This also matches the story: *potent but insoluble and cardiotoxic; keep the
potency, fix the liabilities.*

### 3. The specialist architecture does not beat one generic agent on the front

`python -m scripts.ablation --seeds 12 --rounds 5`, surrogate oracles:

| arm | hypervolume | final affinity | floor held |
|---|---|---|---|
| specialists (4 agents, veto + floor) | 0.313 ± 0.046 | 0.67 | **12/12** |
| one generic agent + same veto + floor | 0.313 ± 0.046 | 0.67 | **12/12** |
| one generic agent, no constraints | 0.349 ± 0.029 | 0.45 | **0/12** |

Two things fall out of this, and both are worth more than the claim they replace:

**The four-agent deliberation contributes nothing measurable to the search.**
Specialists and one-agent-with-the-same-constraints are identical. That is not
a bug: the orchestrator's pick is a deterministic function of the scores, so
the specialists' arguments are narration over a weighted-sum optimiser.

**What the architecture does buy is constraint satisfaction.** The
unconstrained generalist reaches a *better* front — by quietly spending the
potency that made the seed a lead, ending at affinity 0.45, below the floor in
12 runs out of 12. The specialist arm never does. Safety's veto is structural:
it removes molecules rather than weighting them, and nothing downstream can
trade it away.

So the honest claim is **not** "specialists find a better front". It is: *a
single agent asked to balance five properties will silently sacrifice the hard
one to improve its average, and a veto-holding specialist will not.* That is a
real, measured property of the architecture, and it is the thing hERG vetoes
exist for in actual drug programmes.

We also tried wiring the agents' objections back into the edit engine, so
deliberation would steer the chemistry. The obvious rule — propose edits on
whatever axis an agent objected about — is measurably **worse** than ignoring
the agents entirely (0.263 vs 0.301). An objecting agent names the axis it is
already defending; steering there spends proposals on the property not being
lost. See `FEEDBACK_RULE` in `core/loop.py` for all four rules and their
numbers.

---

## Layout

| path | what |
|---|---|
| `core/contract.py` | the interface everything codes against; `normalize`, `is_valid` |
| `core/oracles.py` | `score()` — five axes, cached, provenance-stamped |
| `core/admet.py` | XGBoost backend and the labelled surrogate |
| `core/edits.py` | `propose()` — 19 SMARTS transforms, LLM tier on top |
| `core/agents.py` | three specialists, the veto, the orchestrator |
| `core/loop.py` | the round loop, Pareto front, hypervolume, ablation arms |
| `frontend/app.py` | Streamlit demo; replays `demo_run.json` |
| `frontend/molecule3d.py` | offline RDKit conformer viewer and MCS edit highlighting |
| `frontend/science.py` | candidate explanations and wet-lab assay suggestions |
| `core/drugs.py` | scientist-facing compound lookup and target rationale |
| `core/pubchem.py` | PubChem autocomplete and structure lookup |
| `frontend/plots.py` | headless figures — the cut-list fallback |
| `scripts/verify_models.py` | **the gate.** Run before trusting the models |
| `scripts/ablation.py` | the three-arm comparison above |
| `fixtures.json` | five real molecules in contract format |

## Configuration

| variable | effect |
|---|---|
| `ANTHROPIC_API_KEY` | absent → deterministic agents and mutation-only edits |
| `AGENT_LLM_PROVIDER` | `anthropic` (default) or `bedrock` |
| `BEDROCK_MODEL_ID` | Converse-compatible Bedrock model ID |
| `AWS_REGION` | Bedrock region; defaults to `us-east-1` |
| `AGENT_LLM=off` | force deterministic agents even with a key |
| `EDIT_ENGINE=mutation` | force SMARTS mutations, skip the LLM tier |
| `ADMET_BACKEND` | `xgboost` (fail if absent) or `surrogate` (force) |
| `AFFINITY_ORACLE` | `tdc` or `similarity` |
| `ADMET_MODELS_DIR` | where the inherited models live (default `models/`) |
| `ADMET_FEATURIZER` | `module:function` override for featurization |
| `FEEDBACK_RULE` | `weakest` (default), `none`, `disputed`, `contested-weak` |

## What this isn't

The ADMET models are trained on scaffold splits, so they were tested on novel
chemistry. The agents generate molecules further out of distribution than that:
**the predictions get less reliable exactly as the optimisation gets more
interesting**, and nothing in this loop knows when it has walked off the
training manifold.

The similarity fallbacks measure resemblance, not activity. DRD2 fallback mode
compares against a known-ligand panel. Other targets compare against the
selected starting compound to discourage scaffold drift. Neither establishes
binding, selectivity, or functional potency.

There is no docking, no free-energy calculation, and no synthesis check beyond
an SA score. The real version closes the loop on in vitro assay data, and the
hard problem becomes sample efficiency when every data point costs a few
hundred dollars and four weeks.
