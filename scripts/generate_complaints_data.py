"""
generate_complaints_data.py
---------------------------
Generates synthetic hostel complaint records and inserts them into hostel.db.

Usage:
    python scripts/generate_complaints_data.py          # inserts 500 records (default)
    python scripts/generate_complaints_data.py 100      # inserts 100 records

Data generation logic (rule-based, mirrors realistic hostel scenarios):
    - electricity complaints OR rooms with past_complaints > 6  → 24–72 h
      Rationale: electrical faults need licensed technicians who aren't always
      on-site; high-complaint rooms likely have recurring structural issues.
    - plumbing complaints                                        → 12–36 h
      Rationale: plumbers available during business hours; parts sometimes needed.
    - cleanliness complaints                                     → 2–12 h
      Rationale: housekeeping staff resolve these within a single shift.

No real student data is used. The random seed is NOT fixed here so that each
run produces genuinely new records — this is intentional for data drift
simulation (running this script repeatedly lets us observe how the model
adapts as distribution shifts over time).
"""

import sqlite3
import random
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database", "hostel.db")

NUM_RECORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 500

categories  = ["plumbing", "electricity", "cleanliness"]
days        = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"]

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

inserted = 0
for _ in range(NUM_RECORDS):
    category            = random.choice(categories)
    hostel_age          = random.randint(1, 30)
    floor_number        = random.randint(0, 5)
    room_capacity       = random.choice([1, 2, 3])
    past_complaints     = random.randint(0, 10)
    day                 = random.choice(days)
    past_resolution_avg = random.randint(2, 48)

    # Rule-based resolution time — the signal the model learns from
    if category == "electricity" or past_complaints > 6:
        resolution_time = random.randint(24, 72)
    elif category == "plumbing":
        resolution_time = random.randint(12, 36)
    else:
        resolution_time = random.randint(2, 12)

    cursor.execute("""
        INSERT INTO complaints (
            complaint_category, hostel_age, floor_number, room_capacity,
            past_complaints, day_of_week, past_resolution_avg,
            resolution_time_hours
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (category, hostel_age, floor_number, room_capacity,
          past_complaints, day, past_resolution_avg, resolution_time))
    inserted += 1

conn.commit()

# Report current total so the dashboard's retrain trigger can be verified
cursor.execute("SELECT COUNT(*) FROM complaints")
total = cursor.fetchone()[0]
conn.close()

print(f"Inserted {inserted} new records.")
print(f"Total records in database: {total}")
print("Run train_model.py to retrain if total has grown by 50+ since last training.")