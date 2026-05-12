import csv
import os
from datetime import datetime
from app.constants import CHEATING_DESCRIPTIONS

def init_csv(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["student_id", "session_id", "timestamp", "cheating_status"])

def log_violation(filepath: str, student_id: str, session_id: str, violation_type: str):
    status = CHEATING_DESCRIPTIONS.get(violation_type, violation_type)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            student_id,
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status
        ])