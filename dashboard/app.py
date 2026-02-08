import streamlit as st
import pandas as pd
import joblib
import json
import os
import glob
import sqlite3
import subprocess
import matplotlib.pyplot as plt


# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "database", "hostel.db")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "scripts", "train_model.py")


# Load latest model and metadata
model_files = glob.glob(os.path.join(MODELS_DIR, "model_v*.pkl"))

if not model_files:
    st.error("❌ No trained model found. Please run training first.")
    st.stop()

latest_model_path = max(
    model_files,
    key=lambda x: int(os.path.basename(x).split("_v")[1].split(".pkl")[0])
)

model_version = os.path.basename(latest_model_path).split("_v")[1].split(".pkl")[0]
metadata_path = os.path.join(
    MODELS_DIR, f"model_v{model_version}_metadata.json"
)

model = joblib.load(latest_model_path)

with open(metadata_path, "r") as f:
    metadata = json.load(f)


# Database functions
def load_all_complaints():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM complaints", conn)
    conn.close()
    return df

def get_complaint_count():
    conn = sqlite3.connect(DB_PATH)
    count = pd.read_sql(
        "SELECT COUNT(*) AS c FROM complaints", conn
    )["c"][0]
    conn.close()
    return count

def insert_complaint(row):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO complaints (
            complaint_category,
            hostel_age,
            floor_number,
            room_capacity,
            past_complaints,
            day_of_week,
            past_resolution_avg,
            resolution_time_hours
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, row)
    conn.commit()
    conn.close()


# Global complaint count for model status page

current_count = get_complaint_count()
new_complaints = current_count - metadata["complaints_at_training"]


# Page configuration
st.set_page_config(
    page_title="Hostel Complaint ML System",
    layout="centered"
)

st.title("🏠 Hostel Complaint Management System")


# Sidebar navigation
page = st.sidebar.radio(
    "📂 Navigate",
    ["📌 Model Status", "📊 Analytics", "➕ Add Complaint", "🔮 Prediction"]
)


# Page 1: Model Status
if page == "📌 Model Status":
    st.subheader("📌 Active Model Information")

    st.write(f"**Model File:** `{metadata['model_file']}`")
    st.write(f"**Model Version:** v{metadata['model_version']}")
    st.write(f"**Model Type:** {metadata['best_model']}")
    st.write(f"**MAE:** {metadata['mae']} hours")
    st.write(f"**R² Score:** {metadata['r2_score']}")
    st.write(f"**Trained On:** {metadata['trained_on']}")
    st.write(f"**Complaints at Training:** {metadata['complaints_at_training']}")
    st.write(f"**Current Complaints:** {current_count}")
    st.write(f"**New Complaints Since Training:** {new_complaints}")

    if new_complaints >= 50:
        st.warning(
            f"⚠️ {new_complaints} new complaints detected. Retraining recommended."
        )

        if st.button("🔁 Retrain Model Now"):
            with st.spinner("Retraining model..."):
                subprocess.run(["python", TRAIN_SCRIPT], check=True)
            st.success("✅ Model retrained successfully")
            st.rerun()


# Page 2: Analytics
elif page == "📊 Analytics":
    st.subheader("📊 Complaint Analytics")

    df = load_all_complaints()

# Complaint Category Distribution
    st.markdown("### 🧾 Complaint Category Distribution")
    fig1, ax1 = plt.subplots()
    df["complaint_category"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Number of Complaints")
    st.pyplot(fig1)

# Avg Resolution Time by Category
    st.markdown("### ⏱️ Average Resolution Time by Category")
    fig2, ax2 = plt.subplots()
    df.groupby("complaint_category")["resolution_time_hours"].mean().plot(
        kind="bar", ax=ax2
    )
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Avg Resolution Time (hours)")
    st.pyplot(fig2)

# Complaints by Day of Week
    st.markdown("### 📅 Complaints by Day of Week")
    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    fig3, ax3 = plt.subplots()
    df["day_of_week"].value_counts().reindex(day_order).plot(
        kind="bar", ax=ax3
    )
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Number of Complaints")
    st.pyplot(fig3)


# Page 3: Add Complaint

elif page == "➕ Add Complaint":
    st.subheader("➕ Add New Complaint (Ground Truth)")

    with st.form("add_complaint_form"):
        category = st.selectbox(
            "Complaint Category",
            ["plumbing", "electricity", "cleanliness"]
        )
        hostel_age = st.number_input("Hostel Age (years)", 0, 50, 10)
        floor = st.number_input("Floor Number", 0, 10, 2)
        capacity = st.selectbox("Room Capacity", [1, 2, 3])
        past = st.number_input("Past Complaints from Room", 0, 20, 3)
        day = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
        )
        avg_time = st.number_input(
            "Average Past Resolution Time (hours)", 1, 100, 18
        )
        actual_time = st.number_input(
            "Actual Resolution Time (hours)", 1, 200, 24
        )

        submitted = st.form_submit_button("📥 Save Complaint")

    if submitted:
        insert_complaint((
            category,
            hostel_age,
            floor,
            capacity,
            past,
            day,
            avg_time,
            actual_time
        ))
        st.success("✅ Complaint added successfully")
        st.rerun()


# Page 4: Prediction

elif page == "🔮 Prediction":
    st.subheader("🔮 Predict Resolution Time")

    p_category = st.selectbox(
        "Complaint Category",
        ["plumbing", "electricity", "cleanliness"]
    )
    p_hostel_age = st.number_input("Hostel Age", 0, 50, 10)
    p_floor = st.number_input("Floor Number", 0, 10, 2)
    p_capacity = st.selectbox("Room Capacity", [1, 2, 3])
    p_past = st.number_input("Past Complaints", 0, 20, 3)
    p_day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )
    p_avg = st.number_input(
        "Past Avg Resolution Time (hours)", 1, 100, 18
    )

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([{
            "complaint_category": p_category,
            "hostel_age": p_hostel_age,
            "floor_number": p_floor,
            "room_capacity": p_capacity,
            "past_complaints": p_past,
            "day_of_week": p_day,
            "past_resolution_avg": p_avg
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"🕒 Estimated Resolution Time: **{prediction:.2f} hours**")