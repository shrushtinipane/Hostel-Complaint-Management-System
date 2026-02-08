import sqlite3
import os

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "hostel.db")

os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_category TEXT,
    hostel_age INTEGER,
    floor_number INTEGER,
    room_capacity INTEGER,
    past_complaints INTEGER,
    day_of_week TEXT,
    past_resolution_avg INTEGER,
    resolution_time_hours INTEGER
)
""")

conn.commit()
conn.close()

print("Database and table created successfully")