"""
train_model.py
--------------
Benchmarks 22 regression models on the complaints dataset, auto-selects
the best by MAE on a held-out 20% test split, and saves a versioned
.pkl + metadata JSON.

Pipeline design decisions:
    1. ColumnTransformer splits features into two groups:
       - categorical (complaint_category, day_of_week) → OneHotEncoder
         handle_unknown="ignore" so new categories from future data don't crash
         prediction; sparse_output=False for sklearn compatibility
       - numerical (5 features) → passthrough (no scaling needed for tree models
         which are scale-invariant)

    2. SGDRegressor and PassiveAggressiveRegressor are sensitive to unscaled
       features. They get their own pipeline branch with StandardScaler applied
       to the numerical block so they compete fairly.

    3. Each model is wrapped in a full sklearn Pipeline so preprocessing is
       applied consistently at both fit and predict time — zero data leakage.

    4. Selection criterion: lowest MAE on the test set.
       MAE was chosen over RMSE because resolution time errors are roughly
       linear in cost (being 10h wrong costs ~10× more than being 1h wrong),
       and MAE is less sensitive to the rare extreme outliers in the data.

    5. Versioning: new run always increments from the highest existing version.
"""

import sqlite3
import pandas as pd
import os
import json
import joblib
import glob
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Linear models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, SGDRegressor,
    PassiveAggressiveRegressor
)
# Tree-based
from sklearn.tree import DecisionTreeRegressor
# Ensemble
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor
)
# Support vector
from sklearn.svm import SVR, LinearSVR
# Neighbors
from sklearn.neighbors import KNeighborsRegressor
# Kernel
from sklearn.kernel_ridge import KernelRidge
# Gaussian process
from sklearn.gaussian_process import GaussianProcessRegressor
# Third-party boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, "database", "hostel.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Version detection ───────────────────────────────────────────────────────
existing = glob.glob(os.path.join(MODELS_DIR, "model_v*.pkl"))
next_version = 1 if not existing else max(
    int(os.path.basename(f).split("_v")[1].split(".pkl")[0])
    for f in existing
) + 1

MODEL_PATH    = os.path.join(MODELS_DIR, f"model_v{next_version}.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, f"model_v{next_version}_metadata.json")

# ── Load data ───────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM complaints", conn)
conn.close()

total_records = len(df)
X = df.drop(columns=["complaint_id", "resolution_time_hours"])
y = df["resolution_time_hours"]

categorical_features = ["complaint_category", "day_of_week"]
numerical_features   = [
    "hostel_age", "floor_number", "room_capacity",
    "past_complaints", "past_resolution_avg"
]

# ── Preprocessors ───────────────────────────────────────────────────────────
# Standard preprocessor: OHE for categoricals, passthrough for numerics
# Used by all models except SGD and PAR which need scaled numerics
preprocessor_standard = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", "passthrough", numerical_features)
])

# Scaled preprocessor: OHE for categoricals, StandardScaler for numerics
# SGDRegressor and PassiveAggressiveRegressor are gradient-based and blow up
# when feature magnitudes differ by orders of magnitude (e.g. hostel_age 1–30
# vs past_resolution_avg 2–48). Scaling fixes this.
preprocessor_scaled = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model registry ──────────────────────────────────────────────────────────
# Tuple format: (regressor_instance, use_scaled_preprocessor)
models = {
    "LinearRegression":          (LinearRegression(),                              False),
    "Ridge":                     (Ridge(),                                          False),
    "Lasso":                     (Lasso(),                                          False),
    "ElasticNet":                (ElasticNet(),                                     False),
    "BayesianRidge":             (BayesianRidge(),                                  False),
    "HuberRegressor":            (HuberRegressor(),                                 False),
    "SGDRegressor":              (SGDRegressor(random_state=42, max_iter=1000),      True),
    "PassiveAggressiveRegressor":(PassiveAggressiveRegressor(random_state=42),       True),
    "DecisionTree":              (DecisionTreeRegressor(random_state=42),            False),
    "RandomForest":              (RandomForestRegressor(random_state=42),            False),
    "ExtraTrees":                (ExtraTreesRegressor(random_state=42),              False),
    "GradientBoosting":          (GradientBoostingRegressor(random_state=42),        False),
    "HistGradientBoosting":      (HistGradientBoostingRegressor(random_state=42),    False),
    "AdaBoost":                  (AdaBoostRegressor(random_state=42),                False),
    "SVR":                       (SVR(),                                             False),
    "LinearSVR":                 (LinearSVR(),                                       False),
    "KNN":                       (KNeighborsRegressor(),                             False),
    "KernelRidge":               (KernelRidge(),                                     False),
    "GaussianProcess":           (GaussianProcessRegressor(),                        False),
    "XGBoost":                   (XGBRegressor(random_state=42, verbosity=0),        False),
    "LightGBM":                  (LGBMRegressor(random_state=42, verbose=-1),        False),
    "CatBoost":                  (CatBoostRegressor(verbose=0, random_state=42),     False),
}

# ── Training loop ────────────────────────────────────────────────────────────
best_model, best_model_name = None, None
best_mae, best_r2           = float("inf"), None
all_results                 = {}

print(f"\n{'='*58}")
print(f"  TRAINING {len(models)} MODELS ON {total_records} RECORDS  (v{next_version})")
print(f"{'='*58}")

for name, (regressor, use_scaled) in models.items():
    try:
        prep = preprocessor_scaled if use_scaled else preprocessor_standard
        pipeline = Pipeline(steps=[
            ("preprocessor", prep),
            ("regressor",    regressor)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)

        scaled_tag = " [scaled]" if use_scaled else ""
        all_results[name] = {"mae": round(mae, 4), "r2": round(r2, 4)}
        print(f"  OK {name:<35}{scaled_tag:<10}  MAE={mae:.2f}  R2={r2:.2f}")

        if mae < best_mae:
            best_mae, best_r2   = mae, r2
            best_model          = pipeline
            best_model_name     = name

    except Exception as e:
        print(f"  FAIL {name:<35}  FAILED: {e}")
        all_results[name] = {"mae": None, "r2": None, "error": str(e)}

# ── Save ─────────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_PATH)

metadata = {
    "model_file":             f"model_v{next_version}.pkl",
    "model_version":          next_version,
    "best_model":             best_model_name,
    "mae":                    round(best_mae, 2),
    "r2_score":               round(best_r2, 2),
    "trained_on":             datetime.now().isoformat(),
    "complaints_at_training": total_records,
    "all_model_results":      all_results,
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n{'='*58}")
print(f"  WINNER : {best_model_name}")
print(f"  MAE    : {best_mae:.2f}   R² : {best_r2:.2f}")
print(f"  Saved  : model_v{next_version}.pkl")
print(f"{'='*58}\n")