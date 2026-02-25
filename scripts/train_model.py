import sqlite3
import pandas as pd
import os
import json
import joblib
import glob
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Linear Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, SGDRegressor,
    PassiveAggressiveRegressor
)
# Tree-Based
from sklearn.tree import DecisionTreeRegressor
# Ensemble
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor
)
# Support Vector
from sklearn.svm import SVR, LinearSVR
# Neighbors
from sklearn.neighbors import KNeighborsRegressor
# Kernel
from sklearn.kernel_ridge import KernelRidge
# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
# Boosting libraries
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "hostel.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

existing_models = glob.glob(os.path.join(MODELS_DIR, "model_v*.pkl"))

if not existing_models:
    next_version = 1
else:
    versions = [
        int(os.path.basename(f).split("_v")[1].split(".pkl")[0])
        for f in existing_models
    ]
    next_version = max(versions) + 1

MODEL_PATH = os.path.join(MODELS_DIR, f"model_v{next_version}.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, f"model_v{next_version}_metadata.json")

# Load data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM complaints", conn)
conn.close()

total_records = len(df)

X = df.drop(columns=["complaint_id", "resolution_time_hours"])
y = df["resolution_time_hours"]

categorical_features = ["complaint_category", "day_of_week"]
numerical_features = [
    "hostel_age", "floor_number", "room_capacity",
    "past_complaints", "past_resolution_avg"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    # Linear Models
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "BayesianRidge": BayesianRidge(),
    "HuberRegressor": HuberRegressor(),
    "SGDRegressor": SGDRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    # Tree-Based Models
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    # Ensemble Tree Models
    "RandomForest": RandomForestRegressor(random_state=42),
    "ExtraTrees": ExtraTreesRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    # Support Vector
    "SVR": SVR(),
    "LinearSVR": LinearSVR(),
    # Neighbors
    "KNN": KNeighborsRegressor(),
    # Kernel Methods
    "KernelRidge": KernelRidge(),
    # Gaussian Process
    "GaussianProcess": GaussianProcessRegressor(),
    # Industry Boosting Libraries
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

best_model = None
best_model_name = None
best_mae = float("inf")
best_r2 = None
all_results = {}

print(f"\n{'='*55}")
print(f"  TRAINING {len(models)} MODELS ON {total_records} RECORDS")
print(f"{'='*55}")

for name, regressor in models.items():
    try:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        all_results[name] = {"mae": round(mae, 4), "r2": round(r2, 4)}
        print(f"  ✓ {name:<30} MAE={mae:.2f}  R²={r2:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_r2 = r2
            best_model = pipeline
            best_model_name = name

    except Exception as e:
        print(f"  ✗ {name:<30} FAILED: {e}")
        all_results[name] = {"mae": None, "r2": None, "error": str(e)}

# Save best model
joblib.dump(best_model, MODEL_PATH)

now = datetime.now()

metadata = {
    "model_file": f"model_v{next_version}.pkl",
    "model_version": next_version,
    "best_model": best_model_name,
    "mae": round(best_mae, 2),
    "r2_score": round(best_r2, 2),
    "trained_on": now.isoformat(),
    "complaints_at_training": total_records,
    "all_model_results": all_results
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n{'='*55}")
print(f"  WINNER: {best_model_name}")
print(f"  MAE: {best_mae:.2f}   R²: {best_r2:.2f}")
print(f"  Saved: model_v{next_version}.pkl")
print(f"{'='*55}\n")