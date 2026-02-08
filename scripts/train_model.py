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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
METADATA_PATH = os.path.join(
    MODELS_DIR, f"model_v{next_version}_metadata.json"
)

# Load data from database
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM complaints", conn)
conn.close()

total_records = len(df)

# Feature and target
X = df.drop(columns=["complaint_id", "resolution_time_hours"])
y = df["resolution_time_hours"]

categorical_features = ["complaint_category", "day_of_week"]
numerical_features = [
    "hostel_age",
    "floor_number",
    "room_capacity",
    "past_complaints",
    "past_resolution_avg"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_model_name = None
best_mae = float("inf")
best_r2 = None

# Train and evaluate models
for name, regressor in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")

    if mae < best_mae:
        best_mae = mae
        best_r2 = r2
        best_model = pipeline
        best_model_name = name

# Save best model and metadata
joblib.dump(best_model, MODEL_PATH)

now = datetime.now()

metadata = {
    "model_file": f"model_v{next_version}.pkl",
    "model_version": next_version,
    "best_model": best_model_name,
    "mae": round(best_mae, 2),
    "r2_score": round(best_r2, 2),
    "trained_on": now.isoformat(),
    "complaints_at_training": total_records
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nTRAINING COMPLETE")
print(f"Saved model: model_v{next_version}.pkl")
print(f"Saved metadata: model_v{next_version}_metadata.json")