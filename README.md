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
│   ├── setup_database.py            # Creates the complaints table schema
│   ├── generate_complaints_data.py  # Generates 500 synthetic complaint records
│   └── train_model.py               # Trains 22 models, auto-selects best by MAE
│
├── models/
│   ├── model_v1.pkl                 # Serialised model — version 1
│   ├── model_v1_metadata.json       # Metadata for version 1
│   ├── model_v2.pkl                 # Serialised model — version 2 (current)
│   └── model_v2_metadata.json       # Metadata for version 2
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

Creates the `hostel.db` SQLite database with the `complaints` table.

### 4. Generate synthetic training data

```bash
python scripts/generate_complaints_data.py
```

Inserts **500 synthetic complaint records** into the database using rule-based logic that mirrors realistic hostel scenarios.

### 5. Train the models

```bash
python scripts/train_model.py
```

Trains all 22 regression models, evaluates each on MAE and R², and saves the best-performing model as a versioned `.pkl` file with a corresponding metadata JSON.

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
└── passthrough
      └── hostel_age, floor_number, room_capacity,
          past_complaints, past_resolution_avg
```

Each model is wrapped in a scikit-learn `Pipeline` so preprocessing is applied consistently at both train and predict time — no data leakage.

### Selection Criterion

The model with the **lowest Mean Absolute Error (MAE)** on a held-out test set (20% split, `random_state=42`) is automatically selected and saved as the deployed model.

---

## 📊 Current Model Performance (v2)

Trained on **572 records** on 25 Feb 2026.

| Metric | Value |
|---|---|
| **Champion** | Random Forest |
| **MAE** | 7.92 hours |
| **R² Score** | 0.75 |

### Full Leaderboard (v2 training run)

| Rank | Model | MAE | R² |
|---|---|---|---|
| 🥇 1 | RandomForest | 7.92 | 0.75 |
| 2 | XGBoost | 7.94 | 0.70 |
| 3 | ExtraTrees | 8.20 | 0.70 |
| 4 | GradientBoosting | 8.25 | 0.75 |
| 5 | CatBoost | 8.30 | 0.73 |
| 6 | HistGradientBoosting | 8.37 | 0.75 |
| 7 | LightGBM | 8.37 | 0.75 |
| 8 | AdaBoost | 8.96 | 0.74 |
| 9 | DecisionTree | 9.93 | 0.51 |
| 10 | HuberRegressor | 12.88 | 0.51 |
| 11 | LinearSVR | 13.03 | 0.49 |
| 12 | LinearRegression | 13.34 | 0.51 |
| 13 | KernelRidge | 13.39 | 0.50 |
| 14 | Ridge | 13.36 | 0.51 |
| 15 | BayesianRidge | 13.51 | 0.50 |
| 16 | Lasso | 14.19 | 0.46 |
| 17 | ElasticNet | 15.90 | 0.32 |
| 18 | KNN | 17.14 | 0.13 |
| 19 | SVR | 19.26 | 0.06 |
| 20 | PassiveAggressiveRegressor | 22.71 | -0.39 |
| 21 | GaussianProcess | 28.05 | -1.81 |
| 22 | SGDRegressor | 1931.57 | -10900.31 |

> SGDRegressor and PassiveAggressiveRegressor perform poorly without feature scaling — expected behaviour, noted for future preprocessing improvements.

---

## 📦 Model Versioning & Metadata

Every training run produces two files automatically:

- `model_vN.pkl` — the serialised best-performing pipeline
- `model_vN_metadata.json` — full training record

Example (`model_v2_metadata.json`):

```json
{
    "model_file": "model_v2.pkl",
    "model_version": 2,
    "best_model": "RandomForest",
    "mae": 7.92,
    "r2_score": 0.75,
    "trained_on": "2026-02-25T16:29:44",
    "complaints_at_training": 572,
    "all_model_results": { ... }
}
```

Versioning is automatic — each new training run increments the version number. The dashboard always loads the highest-versioned model file.

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

### ⚡ Retrain from Anywhere

A **Retrain Model** button is pinned in the sidebar and available from every page at all times — no need to navigate away or run scripts manually.

---

## 🔄 Model Lifecycle & Retraining

| Trigger | Action |
|---|---|
| 50+ new complaints since last training | ⚠️ Warning shown on dashboard; retrain recommended |
| Manual override | 🔁 Retrain button available at all times in sidebar and Model Intelligence page |
| Post-retrain | New versioned `.pkl` + metadata JSON saved; dashboard auto-reloads with new model |

As real complaints accumulate through the Add Complaint form, the model continuously improves by learning from ground-truth resolution times.

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
- SGDRegressor and PassiveAggressiveRegressor are sensitive to unscaled features; adding a `StandardScaler` step for these models in a future version would likely improve their rankings significantly.
- Model files can grow large over multiple retraining cycles. Only the latest version is needed in production; older `.pkl` files can be safely archived or deleted.