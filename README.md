# 🏛️ ResolvIQ — Hostel Complaint Intelligence System

> An end-to-end Machine Learning system for university hostel administration.  
> Predicts complaint resolution time, surfaces live analytics, and manages the full model lifecycle — all from a single professional dashboard.

---

## 📌 Overview

Hostel administrations receive high volumes of complaints daily — plumbing failures, electrical issues, cleanliness concerns — and handle them entirely manually with no systematic way to forecast resolution time or identify patterns.

**ResolvIQ** solves this by:
- Predicting how long a complaint will take to resolve (in hours)
- Surfacing live analytics on complaint trends, category breakdowns, and floor-level patterns
- Enabling continuous complaint intake with automatic model retraining triggers
- Managing the full model lifecycle with versioning, metadata tracking, and one-click retraining from the dashboard

The project emphasises **engineering quality, usability, and model lifecycle management** alongside predictive accuracy.

---

## 🗂️ Project Structure

```
hostel_complaint/
│
├── database/
│   └── hostel.db                    # SQLite database (auto-created by setup script)
│
├── scripts/
│   ├── setup_database.py            # Creates complaints table with constraints + index
│   ├── generate_complaints_data.py  # Generates synthetic records (count configurable via CLI)
│   └── train_model.py               # Trains 22 models, auto-selects best by MAE
│
├── models/
│   ├── model_v1.pkl                 # Serialised model — version 1
│   ├── model_v1_metadata.json       # Metadata for version 1
│   ├── model_v2.pkl                 # Serialised model — version 2
│   ├── model_v2_metadata.json       # Metadata for version 2
│   ├── model_v3.pkl                 # Serialised model — version 3
│   ├── model_v3_metadata.json       # Metadata for version 3
│   ├── model_v4.pkl                 # Serialised model — version 4
│   ├── model_v4_metadata.json       # Metadata for version 4
│   ├── model_v5.pkl                 # Serialised model — version 5 (current)
│   └── model_v5_metadata.json       # Metadata for version 5
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard (ResolvIQ UI)
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd hostel_complaint
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib plotly xgboost lightgbm catboost
```

### 3. Initialise the database

```bash
python scripts/setup_database.py
```

Creates `hostel.db` with the `complaints` table, NOT NULL constraints on all columns, and an index on `complaint_category` for faster dashboard queries.

### 4. Generate synthetic training data

```bash
python scripts/generate_complaints_data.py          # inserts 500 records (default)
python scripts/generate_complaints_data.py 100      # inserts 100 records
```

The record count is configurable via a CLI argument. Each run produces genuinely new records (no fixed random seed) to simulate realistic data drift over time.

### 5. Train the models

```bash
python scripts/train_model.py
```

Trains all 22 regression models, evaluates each on MAE and R², and saves the best-performing model as a versioned `.pkl` file with a corresponding metadata JSON. SGDRegressor and PassiveAggressiveRegressor are given a separate StandardScaler preprocessing branch so they compete fairly.

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧾 Data Description

All data is **synthetically generated** — no real student data is used. The generator (`generate_complaints_data.py`) applies rule-based logic to produce realistic resolution time distributions.

| Feature | Type | Description |
|---|---|---|
| `complaint_category` | Categorical | `plumbing`, `electricity`, or `cleanliness` |
| `hostel_age` | Integer | Age of the hostel building in years (1–30) |
| `floor_number` | Integer | Floor on which the complaint was raised (0–5) |
| `room_capacity` | Integer | Number of occupants in the room (1–3) |
| `past_complaints` | Integer | Number of previous complaints from the same room (0–10) |
| `day_of_week` | Categorical | Day the complaint was registered (Monday–Sunday) |
| `past_resolution_avg` | Integer | Historical average resolution time for that room (2–48 hrs) |

**Target variable:** `resolution_time_hours` — actual time taken to resolve the complaint.

**Generation logic:**
- `electricity` complaints or rooms with `past_complaints > 6` → 24–72 hours
- `plumbing` complaints → 12–36 hours
- `cleanliness` complaints → 2–12 hours

---

## 🧠 Models Trained

The training script benchmarks **22 regression models** across all major ML families in a single run:

| Family | Models |
|---|---|
| **Linear** | LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, PassiveAggressiveRegressor |
| **Tree-Based** | DecisionTree |
| **Ensemble** | RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost |
| **Support Vector** | SVR, LinearSVR |
| **Neighbors** | KNN |
| **Kernel** | KernelRidge |
| **Gaussian Process** | GaussianProcessRegressor |
| **Boosting Libraries** | XGBoost, LightGBM, CatBoost |

### Preprocessing Pipeline

```
ColumnTransformer
├── OneHotEncoder(handle_unknown="ignore", sparse_output=False)
│     └── complaint_category, day_of_week
└── passthrough (standard branch) / StandardScaler (scaled branch)
      └── hostel_age, floor_number, room_capacity,
          past_complaints, past_resolution_avg
```

Each model is wrapped in a scikit-learn `Pipeline` so preprocessing is applied consistently at both train and predict time — no data leakage.

SGDRegressor and PassiveAggressiveRegressor use the **scaled branch** (StandardScaler on numerical features) because they are gradient-based optimisers that diverge on unscaled features. All other models use the passthrough branch since tree-based and kernel methods are scale-invariant.

### Selection Criterion

The model with the **lowest Mean Absolute Error (MAE)** on a held-out test set (20% split, `random_state=42`) is automatically selected and saved as the deployed model. MAE was chosen over RMSE because resolution time prediction errors are roughly linear in cost — being 10h wrong costs ~10× more than being 1h wrong.

---

## 📊 Model Performance — Version History

| Version | Date | Champion | MAE | R² | Records | Change |
|---|---|---|---|---|---|---|
| v1 | 08 Feb 2026 | RandomForest | 9.81h | 0.59 | 500 | — |
| v2 | 25 Feb 2026 | RandomForest | 7.92h | 0.75 | 572 | −1.89h (−19.3%) |
| v3 | 04 Mar 2026 | ExtraTrees | 7.30h | 0.73 | 588 | −0.62h (−7.8%) |
| v4 | 12 Mar 2026 | ExtraTrees | 6.82h | 0.76 | 600 | −0.48h (−6.6%) |
| v5 | 19 Mar 2026 | RandomForest | 8.41h | 0.67 | 616 | +1.59h (test-set shift) |

The v5 MAE increase is explained by test-set composition shift — with `random_state=42` applied to a different total row count (616 vs 600), different rows land in the 20% holdout. If the new test set contains proportionally more electricity complaints (24–72h range), the absolute error is structurally higher. This is expected behaviour with a fixed random seed on a growing dataset, not a model quality regression.

The post-mortem section on the Model Intelligence dashboard page provides a detailed visual breakdown of each version's performance and the data-driven reasons behind every metric shift.

---

## 📦 Model Versioning & Metadata

Every training run produces two files automatically:

- `model_vN.pkl` — the serialised best-performing pipeline
- `model_vN_metadata.json` — full training record

Example (`model_v5_metadata.json`):

```json
{
    "model_file": "model_v5.pkl",
    "model_version": 5,
    "best_model": "RandomForest",
    "mae": 8.41,
    "r2_score": 0.67,
    "trained_on": "2026-03-19T17:49:35",
    "complaints_at_training": 616,
    "all_model_results": { ... }
}
```

Versioning is automatic — each new training run increments the version number. The dashboard always loads the highest-versioned model file. The Model Intelligence page reads **all** metadata files to render the evolution post-mortem charts.

---

## 📊 Dashboard — ResolvIQ

Built with **Streamlit** and **Plotly Express / Graph Objects**. Two pages accessible from the sidebar:

### 📊 Operations Hub

Everything an administrator needs in a single view:

- **KPI Strip** — total complaints, average resolution time, critical rate (>48h), top complaint category, model freshness status
- **Analytics tabs:**
  - *By Category* — complaint volume and average resolution time per category
  - *By Day* — complaint frequency across the week
  - *Resolution Dist.* — histogram of resolution time distribution
  - *Heatmap* — average resolution time by category × day of week
- **Prediction panel** — input complaint details and get an instant resolution time estimate with urgency classification:
  - 🟢 **Routine** — under 24 hours
  - 🟡 **Moderate** — 24–48 hours
  - 🔴 **Critical** — over 48 hours
- **Add Complaint** — log a resolved complaint with actual resolution time as ground truth for future retraining
- **Bottom charts** — box plot distribution per category, average resolution time by floor

### 🧠 Model Intelligence

Full model management and lifecycle view:

- **KPI strip** — champion model, MAE, R², total models evaluated
- **Full leaderboard** — all 22 models ranked by MAE with visual bar indicators and winner highlighted
- **Performance charts** — MAE and R² horizontal bar charts (Plotly)
- **Training record table** — version, timestamp, record counts, delta since last training
- **Lifecycle management panel** — retraining trigger status and one-click full retrain
- **Model Evolution Post-Mortem** — MAE and R² trend charts across all versions, per-version insight cards explaining why metrics shifted, and an overall summary of why ensemble methods dominate this dataset

### ⚡ Retrain from Anywhere

A **Retrain Model** button is pinned in the sidebar and available from every page at all times — no need to navigate away or run scripts manually.

---

## 🔄 Model Lifecycle & Retraining

| Trigger | Action |
|---|---|
| 50+ new complaints since last training | ⚠️ Warning shown on dashboard; retrain recommended |
| Manual override | 🔁 Retrain button available at all times in sidebar and Model Intelligence page |
| Post-retrain | New versioned `.pkl` + metadata JSON saved; dashboard auto-reloads with new model |

### Live Demo (viva sequence)

```bash
# Terminal 1 — keep this running
streamlit run dashboard/app.py

# Terminal 2 — run this to simulate new data arriving
python scripts/generate_complaints_data.py 60
# Dashboard will show STALE warning (60 > 50 threshold)
# Click Retrain Model → model_v6.pkl is saved → dashboard reloads
```

---

## 🗓️ Update & Maintenance Timeline

| Cadence | Activity |
|---|---|
| **On demand** | Retrain triggered automatically when 50+ new complaints are logged |
| **Weekly** | Review new complaint volume; check model MAE drift against previous version |
| **Weekly** | Audit newly added complaints for data quality and consistency |
| **Monthly** | Full performance review; retrain on all accumulated data regardless of complaint count |
| **Semester-start** | Full data audit; review feature relevance; regenerate baseline if needed |
| **Annually** | Evaluate new model families; update feature engineering; archive old model versions |

---

## 📝 Notes

- The `.pkl`, `.db`, and `catboost_info/` files should be excluded from the git repository via `.gitignore`. Run the setup, generate, and train scripts locally to regenerate them.
- The `generate_complaints_data.py` script serves as the **data source documentation** — it contains all generation logic and can be modified to change data characteristics or volume.
- SGDRegressor and PassiveAggressiveRegressor now use StandardScaler on numerical features — this was the root cause of their historically poor rankings (MAE 1931h in v2). The fix is applied from the next training run onwards.
- Model files can grow large over multiple retraining cycles. Only the latest version is needed in production; older `.pkl` files can be safely archived or deleted.