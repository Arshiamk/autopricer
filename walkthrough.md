# AutoPricer Walkthrough (Repo Overview)

AutoPricer is a production-style pricing system that demonstrates end-to-end profit optimisation (not notebook-only modelling). It generates realistic synthetic vehicle enquiry/auction data (including counterfactual offers), builds a minimal warehouse, trains pricing + conversion models, selects offers via an Expected Value policy, exposes a FastAPI quote endpoint, and provides drift/performance monitoring via Streamlit.

## Why this project exists (commercial intent)

Most "pricing ML" demos stop at predicting a price. In a real buying operation, the problem is policy optimisation: choosing an offer that balances margin vs conversion under uncertainty and downside risk. AutoPricer is structured around that decision layer.

## 1) Phase 0 — System Skeleton & Tooling

- Lean repo setup (Python config, linting, tests, Make targets).
- Docker Compose stack: Postgres + API + Dashboard.
- CI pipeline (GitHub Actions): lint → tests → docker build.

Deliverable: a runnable skeleton where `make lint`, `make test`, and `make serve` are deterministic.

## 2) Phase 1 — Realistic Synthetic Dataset (Counterfactual Offers)

**Key design:** `--offers-per-enquiry` in `data/seed/generate.py`.

### The issue in real pricing data

Many datasets capture only a single offered price per vehicle and whether it was accepted. That makes it difficult for a supervised model to learn offer elasticity reliably, and it limits the ability to optimise pricing policy.

### The solution implemented here

Each vehicle/enquiry is assigned a latent `true_market_value`, then the generator creates multiple counterfactual offer prices against that same latent value. Win probability is sampled via a sigmoid function, producing a smooth, learnable offer→win curve:

`p_win = sigmoid((offer − true_market_value + margin_threshold) / sensitivity)`

This enables training a conversion model that can be used for policy optimisation (not just correlation).

Outputs include: `vehicles.csv`, `regions.csv`, `enquiries.csv`, `sales.csv` (with win labels + realised margins)

## 3) Phases 2 & 3 — Data Foundations + Models + Evaluation

### Warehouse & "single source of truth"

- Loads CSVs into Postgres.
- dbt builds a minimal staging layer and a single wide `mart_training_set` used for training.

### Models

- **Sale-price model:** XGBoost regressor for E(sale_price)
- **Quantile bounds:** q10/q90 bounds derived by split-conformal calibration of prediction ratios on a held-out set (used for tail-risk penalties)
- **Conversion model:** non-linear classifier trained with `offer_price` and an engineered `offer_to_value_ratio` feature; isotonic probability calibration improves decision reliability.

### Evaluation (commercial + ML)

`pipelines/evaluate/evaluate.py` scores the trained models on held-out test splits and writes `reports/model_evaluation.json`:

- MAE (price), AUC/Brier (conversion)
- empirical coverage of the conformal q10 bound

## 4) Phase 4 — Drift Monitoring + Retrain Alerts

AutoPricer includes lightweight monitoring scripts designed around production signals:

- **Feature drift:** PSI tests (threshold > 0.25) and KS tests (p-value < 0.05) on key numeric features (e.g. vehicle age, mileage, offer, damage severity)
- **Performance tracking:** rolling error and profit simulation
- **Alerts:** thresholds trigger a `retrain_required.json` output

This demonstrates "owning performance in the real world", not just training once.

## 5) Phase 5 — Streamlit Dashboard (3 pages)

A simple dashboard provides visual proof of the system:

- **Overview:** win-rate, margin, expected value vs realised outcomes
- **Drift Monitor:** PSI/KS indicators + quick diagnostics
- **Offer Simulator:** calls the live FastAPI `/quote` endpoint and visualises the EV curve + chosen offer

## EV Policy Optimiser (Core Logic)

The recommended offer is chosen via grid search over a valid offer range:

`EV(offer) = P(win∣offer) × (E(sale) − offer − E(costs)) − λ × tail_penalty`

Tail penalty is intentionally not labelled CVaR unless computed explicitly; in this repo it is a documented lower-quantile penalty (e.g., based on q10 bounds).

## How to run locally

```bash
make setup
make generate features train
docker-compose up --build
```

API: `http://localhost:8000/health`  
Dashboard: `http://localhost:8501`

**Test coverage:**

- optimiser unit tests (pure Python)
- API contract tests (FastAPI TestClient)

Run:

```bash
make test
```
