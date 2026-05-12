from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from app.config import MAX_WARNINGS


class ExamState:
    def __init__(
        self,
        student_id: str,
        session_id: str,
        exam_id:    str,
        course_id:  str,
    ):
        self.student_id = student_id
        self.session_id = session_id
        self.exam_id    = exam_id
        self.course_id  = course_id

        self.warnings   = 0
        self.terminated = False

        # final report يتبعت مرة واحدة بس
        self.final_report_sent = False

        self.no_face_start    = None
        self.multi_face_start = None
        self.head_turn_start  = None
        self.phone_start      = None
        self.book_start       = None

        self.last_warning_type     = None
        self.last_message          = "No warnings"
        self.warning_display_until = 0.0
        self.cooldown_until        = 0.0

        self.started_at = datetime.utcnow().isoformat()
        self.ended_at   = None

        self.violations_history: list[dict[str, Any]] = []

    def _build_message(self, violation_type: str) -> str:
        mapping = {
            "NO_FACE"        : f"Warning {self.warnings}/{MAX_WARNINGS}: Face not visible",
            "MULTIPLE_FACES" : f"Warning {self.warnings}/{MAX_WARNINGS}: Multiple faces detected",
            "HEAD_TURN_LEFT" : f"Warning {self.warnings}/{MAX_WARNINGS}: Looking left detected",
            "HEAD_TURN_RIGHT": f"Warning {self.warnings}/{MAX_WARNINGS}: Looking right detected",
            "HEAD_TURN_DOWN" : f"Warning {self.warnings}/{MAX_WARNINGS}: Looking down detected",
            "HEAD_TURN_UP"   : f"Warning {self.warnings}/{MAX_WARNINGS}: Looking up detected",
            "PHONE_DETECTED" : f"Warning {self.warnings}/{MAX_WARNINGS}: Mobile phone detected",
            "BOOK_DETECTED"  : f"Warning {self.warnings}/{MAX_WARNINGS}: Book detected",
            "PERSON_DETECTED": f"Warning {self.warnings}/{MAX_WARNINGS}: Another person detected",
        }
        return mapping.get(violation_type, f"Warning {self.warnings}/{MAX_WARNINGS}: Violation detected")

    def add_warning(self, violation_type: str, log_callback) -> tuple[int, bool]:
        now = time.time()

        if now < self.cooldown_until:
            return self.warnings, False

        self.warnings             += 1
        self.last_warning_type     = violation_type
        self.warning_display_until = now + 2.0
        self.cooldown_until        = now + 3.0
        self.last_message          = self._build_message(violation_type)

        violation_record = {
            "violation_type": violation_type,
            "timestamp"     : datetime.utcnow().isoformat(),
            "warning_number": self.warnings,
            "terminated"    : False,
            "message"       : self.last_message,
        }
        self.violations_history.append(violation_record)

        log_callback(
            student_id     = self.student_id,
            session_id     = self.session_id,
            violation_type = violation_type,
        )

        if self.warnings >= MAX_WARNINGS:
            self.terminated                = True
            self.ended_at                  = datetime.utcnow().isoformat()
            violation_record["terminated"] = True
            self.last_message              = "Exam terminated: maximum warnings exceeded"

        return self.warnings, True

    def end_session(self):
        if self.ended_at is None:
            self.ended_at = datetime.utcnow().isoformat()

    def to_live_dict(self) -> dict[str, Any]:
        return {
            "session_id"    : self.session_id,
            "student_id"    : self.student_id,
            "exam_id"       : self.exam_id,
            "course_id"     : self.course_id,
            "warnings"      : self.warnings,
            "terminated"    : self.terminated,
            "last_violation": self.last_warning_type,
            "last_message"  : self.last_message,
            "status"        : "terminated" if self.terminated
                              else "suspicious" if self.warnings > 0
                              else "ok",
        }

    def to_final_report(self) -> dict[str, Any]:
        return {
            "session_id"        : self.session_id,
            "student_id"        : self.student_id,
            "exam_id"           : self.exam_id,
            "course_id"         : self.course_id,
            "total_warnings"    : self.warnings,
            "terminated"        : self.terminated,
            "started_at"        : self.started_at,
            "ended_at"          : self.ended_at,
            "violations_history": self.violations_history,
        }
