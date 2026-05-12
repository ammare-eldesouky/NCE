import cv2
import asyncio
import websockets
import httpx
import json

BASE_URL = "http://127.0.0.1:8000"
WS_BASE  = "ws://127.0.0.1:8000"

# بيانات الطالب للتجربة
STUDENT_ID = "STU_001"
EXAM_ID    = "EXAM_001"
COURSE_ID  = "COURSE_001"


def start_session() -> str:
    """
    بيعمل POST /session/start ويرجع session_id
    """
    url  = f"{BASE_URL}/session/start"
    body = {
        "student_id": STUDENT_ID,
        "exam_id":    EXAM_ID,
        "course_id":  COURSE_ID,
    }
    response = httpx.post(url, json=body, timeout=5.0)
    data     = response.json()
    print(f"✅ Session started: {data['session_id']}")
    return data["session_id"]


def end_session(session_id: str):
    """
    بيعمل POST /session/end ويطبع التقرير النهائي
    """
    url  = f"{BASE_URL}/session/end"
    body = {"session_id": session_id}
    response = httpx.post(url, json=body, timeout=5.0)
    data     = response.json()
    print("\n📊 Final Report:")
    print(f"   Total Warnings : {data['total_warnings']}")
    print(f"   Terminated     : {data['terminated']}")
    print(f"   Started At     : {data['started_at']}")
    print(f"   Ended At       : {data['ended_at']}")
    print(f"   Violations     :")
    for v in data["violations_history"]:
        print(f"      - {v['violation_type']} | {v['timestamp']} | Warning #{v['warning_number']}")


async def stream_camera(session_id: str):
    """
    بيفتح الكاميرا ويبعت frames للسيرفر عبر WebSocket
    """
    ws_url = f"{WS_BASE}/ws/exam/{session_id}"
    cap    = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print(f"📡 Connecting to WebSocket: {ws_url}")

    async with websockets.connect(ws_url, max_size=10_000_000) as ws:
        print("✅ WebSocket Connected - Press Q to stop\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # ضغط الصورة
            ok, buffer = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                continue

            # بعت الـ frame
            await ws.send(buffer.tobytes())

            try:
                # استقبال الرد
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data     = json.loads(response)

                # طباعة الحالة
                status = f"W:{data['warnings']} | Faces:{data['face_count']} | {data['message']}"
                print(f"📩 {status}")

                # لو الامتحان اتوقف
                if data["terminated"]:
                    print("🛑 Exam Terminated!")
                    break

                # عرض على الشاشة
                cv2.putText(
                    frame, status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

            except asyncio.TimeoutError:
                print("⚠️ No response from server")

            cv2.imshow("NCE Test Client", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Camera stopped")


async def main():
    # 1. ابدأ الجلسة من الباك اند
    session_id = start_session()

    try:
        # 2. ابعت frames عبر WebSocket
        await stream_camera(session_id)
    finally:
        # 3. انهي الجلسة واطبع التقرير
        end_session(session_id)


if __name__ == "__main__":
    asyncio.run(main())
