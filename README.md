# RetainIQ

End-to-end customer churn analytics & prediction system: predicts which customers are about to
churn, explains why (SHAP/LIME), recommends what to do about it (Next-Best-Action), and exposes
it all through a REST API and an interactive dashboard, with drift monitoring and retraining.

Built on the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, ~26.5% churn rate).

See `CLAUDE.md` for full architecture, conventions, and the build-phase tracker.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it at
`data/raw/telco.csv` (data files are git-ignored).

## Status

Phase 0 (scaffold) complete. See `CLAUDE.md` section 14 for the phase tracker.
