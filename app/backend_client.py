from __future__ import annotations

import httpx

from app.config import (
    BACKEND_URL,
    LIVE_VIOLATION_ENDPOINT,
    FINAL_REPORT_ENDPOINT,
)


async def send_live_violation(payload: dict) -> bool:
    url = f"{BACKEND_URL}{LIVE_VIOLATION_ENDPOINT}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"[AI -> Backend] live violation failed: {e}")
        return False


async def send_final_report(payload: dict) -> bool:
    url = f"{BACKEND_URL}{FINAL_REPORT_ENDPOINT}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"[AI -> Backend] final report failed: {e}")
        return False