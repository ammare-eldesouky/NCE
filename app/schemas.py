from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    student_id: str
    exam_id: str
    course_id: str


class EndSessionRequest(BaseModel):
    session_id: str