import sqlite3
import random
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "hostel.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

categories = ["plumbing", "electricity", "cleanliness"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

NUM_RECORDS = 500

for _ in range(NUM_RECORDS):
    category = random.choice(categories)
    hostel_age = random.randint(1, 30)
    floor = random.randint(0, 5)
    room_capacity = random.choice([1, 2, 3])
    past_complaints = random.randint(0, 10)
    day = random.choice(days)
    past_resolution_avg = random.randint(2, 48)

    # realistic rule-based logic
    if category == "electricity" or past_complaints > 6:
        resolution_time = random.randint(24, 72)
    elif category == "plumbing":
        resolution_time = random.randint(12, 36)
    else:
        resolution_time = random.randint(2, 12)

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
    """, (
        category,
        hostel_age,
        floor,
        room_capacity,
        past_complaints,
        day,
        past_resolution_avg,
        resolution_time
    ))

conn.commit()
conn.close()

print("Complaint data generated and inserted into database")