from __future__ import annotations

import asyncio
import uuid
import cv2
import numpy as np

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas        import StartSessionRequest, EndSessionRequest
from app.state          import ExamState
from app.monitor        import analyze_frame
from app.config         import CSV_FILE
from app.logger_csv     import init_csv
from app.backend_client import send_live_violation, send_final_report
from app.vision.face_detection import face_detector
from app.vision.head_pose      import face_mesh

app = FastAPI(title="NCE AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict[str, ExamState]    = {}
session_locks:   dict[str, asyncio.Lock] = {}


# ─── Startup / Shutdown ──────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    init_csv(CSV_FILE)
    print("[NCE AI] Service started")


@app.on_event("shutdown")
def shutdown_event():
    try:
        face_detector.close()
    except Exception:
        pass
    try:
        face_mesh.close()
    except Exception:
        pass


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "NCE AI Service is running"}


@app.get("/health")
def health():
    return {
        "status"         : "ok",
        "active_sessions": len(active_sessions),
    }


# ─── 1. الباك اند يبدأ جلسة ──────────────────────────────────────────────────

@app.post("/session/start")
def start_session(payload: StartSessionRequest):
    session_id = str(uuid.uuid4())

    active_sessions[session_id] = ExamState(
        student_id = payload.student_id,
        session_id = session_id,
        exam_id    = payload.exam_id,
        course_id  = payload.course_id,
    )
    session_locks[session_id] = asyncio.Lock()

    print(f"[Session Start] student={payload.student_id} session={session_id}")

    return {
        "session_id": session_id,
        "student_id": payload.student_id,
        "exam_id"   : payload.exam_id,
        "course_id" : payload.course_id,
        "message"   : "Session created successfully",
    }


# ─── 2. الباك اند ينهي الجلسة ────────────────────────────────────────────────

@app.post("/session/end")
async def end_session(payload: EndSessionRequest):
    state = active_sessions.get(payload.session_id)

    if state is None:
        return {"error": "Session not found"}

    state.end_session()
    report = state.to_final_report()

    if not state.final_report_sent:
        await send_final_report(report)
        state.final_report_sent = True

    del active_sessions[payload.session_id]
    session_locks.pop(payload.session_id, None)

    print(f"[Session End] session={payload.session_id} warnings={state.warnings}")

    return {
        **report,
        "message": "Session ended successfully",
    }


# ─── 3. تقرير لحظي للدكتور المراقب ──────────────────────────────────────────

@app.get("/monitor/live/{exam_id}")
def monitor_live(exam_id: str):
    students = [
        state.to_live_dict()
        for state in active_sessions.values()
        if state.exam_id == exam_id
    ]

    return {
        "exam_id"        : exam_id,
        "total_students" : len(students),
        "active_sessions": students,
    }


# ─── 4. الباك اند يبعت frames عبر HTTP POST ──────────────────────────────────

@app.post("/analyze/{session_id}")
async def analyze(session_id: str, request: Request):
    state = active_sessions.get(session_id)

    if state is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    frame_bytes = await request.body()

    # حماية من frame كبير جداً (أكبر من 2MB)
    if len(frame_bytes) > 2_000_000:
        return JSONResponse(status_code=400, content={"error": "Frame too large"})

    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid frame"})

    # Lock لمنع race condition لو الباك اند بعت requests بسرعة
    async with session_locks[session_id]:
        result = analyze_frame(frame, state)

        # لو warning حقيقي اتضاف → ابعت للباك اند فوراً
        if result.get("new_warning_added", False):
            await send_live_violation({
                "session_id"    : state.session_id,
                "student_id"    : state.student_id,
                "exam_id"       : state.exam_id,
                "course_id"     : state.course_id,
                "violation_type": state.last_warning_type,
                "warning_number": state.warnings,
                "terminated"    : state.terminated,
                "timestamp"     : state.violations_history[-1]["timestamp"],
                "message"       : state.last_message,
            })

        # لو الامتحان اتوقف → ابعت final report مرة واحدة
        if result.get("terminated", False):
            state.end_session()
            if not state.final_report_sent:
                await send_final_report(state.to_final_report())
                state.final_report_sent = True
            print(f"[Terminated] session={session_id}")

    print(f"[FRAME] session={session_id} warnings={state.warnings}")

    return {
        "session_id"       : state.session_id,
        "student_id"       : state.student_id,
        "exam_id"          : state.exam_id,
        "course_id"        : state.course_id,
        "warnings"         : result.get("warnings", 0),
        "terminated"       : result.get("terminated", False),
        "violation_type"   : result.get("violation", None),
        "message"          : result.get("message", ""),
        "face_count"       : result.get("face_count", 0),
        "mean_y"           : result.get("mean_y", 0.0),
        "new_warning_added": result.get("new_warning_added", False),
    }