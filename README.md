# 🏠 Hostel Complaint Management & Resolution Time Prediction System

## 📌 Overview
This project implements an **end-to-end, production-style Machine Learning system** for managing hostel complaints in a university environment.  
The system predicts the **expected resolution time (in hours)** for hostel complaints, provides **data analytics**, supports **continuous data ingestion**, and manages the **complete model lifecycle** including versioning and retraining.

The primary focus of this project is **engineering, usability, and model lifecycle management**, rather than model performance alone.

---

## 🎯 Problem Statement
Hostel administrations receive a large number of complaints related to infrastructure and operations. These complaints are often handled manually without systematic prioritization or forecasting of resolution time.

This system aims to:
- Predict complaint resolution time
- Provide insights into complaint patterns
- Enable data-driven decision making
- Demonstrate a usable ML system beyond notebooks

---

## 🗂️ Project Structure
hostel_complaint_project/
│
├── database/
│ └── hostel.db
│
├── scripts/
│ ├── setup_database.py
│ ├── generate_complaints_data.py
│ └── train_model.py
│
├── models/
│ ├── model_v1.pkl
│ ├── model_v1_metadata.json
│
├── dashboard/
│ └── app.py
│
└── README.md


---

## 🧾 Data Description

### Features
- `complaint_category` – Type of complaint (plumbing, electricity, cleanliness)
- `hostel_age` – Age of hostel building (years)
- `floor_number` – Floor where complaint occurred
- `room_capacity` – Number of occupants in room
- `past_complaints` – Past complaints from the same room
- `day_of_week` – Day when complaint was registered
- `past_resolution_avg` – Average historical resolution time

### Target Variable
- `resolution_time_hours` – Actual time taken to resolve complaint (hours)

All data is **synthetically generated using Python logic** and stored directly in a **SQLite database**.

---

## 🧠 Model Training & Selection
Two regression models are trained and compared:
- Linear Regression
- Random Forest Regressor

### Evaluation
- Models are evaluated using **Mean Absolute Error (MAE)**
- The model with the **lowest MAE** is selected automatically
- Only the **best-performing model** is deployed

---

## 📦 Model Versioning & Metadata
Each training run produces:
- A versioned model file (`model_vX.pkl`)
- A corresponding metadata file (`model_vX_metadata.json`)

Metadata includes:
- Model version
- Model type
- MAE and R² score
- Training timestamp
- Number of complaints used during training

This ensures **full traceability and reproducibility**.

---

## 🔄 Model Lifecycle & Retraining
- The dashboard monitors **new complaints added after the last training**
- When **50 or more new complaints** are detected:
  - A retraining warning is shown
  - A retrain button becomes available
- Retraining:
  - Trains models again on updated data
  - Creates a new versioned model
  - Automatically updates the deployed model

---

## 📊 Dashboard Features
The Streamlit dashboard is divided into **multiple pages**:

### 📌 Model Status
- Active model version and type
- Model performance metrics
- Training timestamp
- Complaint counts and data drift monitoring

### 📊 Analytics
- Complaint category distribution
- Average resolution time by category
- Complaints by day of week  
(Only aggregated bar charts; no scatter/dot plots)

### ➕ Add Complaint
- Append new complaints to the database
- Store **actual resolution time** as ground truth

### 🔮 Prediction
- Predict expected resolution time for new complaints
- Always uses the **latest trained model**