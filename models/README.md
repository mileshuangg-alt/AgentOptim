# models/

This directory contains the three trained XGBoost estimators used by the
optimization loop:

- `solubility_xgb.pkl`: predicted aqueous solubility as LogS
- `bbb_xgb.pkl`: probability of blood-brain barrier penetration
- `herg_xgb.pkl`: probability of hERG blockade

They are copied byte-for-byte from
[`tuhinc5203/admet-property-prediction`](https://github.com/tuhinc5203/admet-property-prediction)
at the commit recorded in `manifest.json`. The manifest also records a SHA-256
digest for every file.

Run the reproducibility gate after installing `requirements.txt`:

    python -m scripts.verify_models

For a direct comparison with a local checkout of the source repository:

    python -m scripts.verify_models --source-repo /path/to/admet-property-prediction

RDKit is a runtime dependency rather than a separate model artifact. It
computes the Morgan fingerprint, six molecular descriptors, and the synthetic
accessibility score. The verification gate checks that the source and local
feature tables are identical.
