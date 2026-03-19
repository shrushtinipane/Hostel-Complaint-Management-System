"""
setup_database.py
-----------------
Creates the SQLite database and complaints table for ResolvIQ.

Schema decisions:
- complaint_id: auto-incrementing PK so each row is uniquely addressable
- complaint_category: TEXT (plumbing / electricity / cleanliness) — kept as raw
  string so the ML pipeline's OneHotEncoder can handle it at train time,
  preventing any leakage between label encoding and the test split
- hostel_age, floor_number, room_capacity, past_complaints,
  past_resolution_avg: INTEGER — all ordinal/numeric features passed through
  the pipeline's passthrough transformer unchanged
- day_of_week: TEXT — categorical, OHE'd alongside complaint_category
- resolution_time_hours: INTEGER — regression target (hours to resolve)
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR   = os.path.join(BASE_DIR, "database")
DB_PATH  = os.path.join(DB_DIR, "hostel.db")

os.makedirs(DB_DIR, exist_ok=True)

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_category    TEXT    NOT NULL,
    hostel_age            INTEGER NOT NULL,
    floor_number          INTEGER NOT NULL,
    room_capacity         INTEGER NOT NULL,
    past_complaints       INTEGER NOT NULL,
    day_of_week           TEXT    NOT NULL,
    past_resolution_avg   INTEGER NOT NULL,
    resolution_time_hours INTEGER NOT NULL
)
""")

# Index on complaint_category speeds up the dashboard's groupby aggregations
cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_category
ON complaints (complaint_category)
""")

conn.commit()
conn.close()

print("Database initialised: hostel.db")
print("Table 'complaints' ready with index on complaint_category.")