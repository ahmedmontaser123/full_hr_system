import time
import logging
from pathlib import Path
from uuid import uuid4
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from schemas import EvaluationSchema
from interview import InterviewSession
from llm import QuestionsGenerator, Evaluator, ClassificationQuestion

logger = logging.getLogger("hr_interview.api")
router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    role: str
    skills: str


class SessionResponse(BaseModel):
    id: str
    role: str
    skills: str
    current_question: Optional[str] = None
    current_category: Optional[str] = None
    question_history: list[str] = []
    results_count: int = 0


class QuestionResponse(BaseModel):
    question: str
    category: str


class AnswerResponse(BaseModel):
    question: str
    category: Optional[str] = None
    transcript: str
    evaluation: EvaluationSchema


class SummaryResponse(BaseModel):
    total_questions: int
    evaluated: int
    final_score: str
    results: list[AnswerResponse] = []


def _get_store(request: Request) -> dict:
    return request.app.state.sessions


def _get_interview_deps(request: Request):
    chains = request.app.state._get_chains()
    transcript = request.app.state._get_transcript()
    return chains, transcript


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(body: CreateSessionRequest, request: Request):
    store = _get_store(request)
    chains, transcript = _get_interview_deps(request)
    session_id = str(uuid4())
    session = InterviewSession(
        transcript=transcript,
        generator=QuestionsGenerator(chains),
        classifier=ClassificationQuestion(chains),
        evaluator=Evaluator(chains),
    )
    store[session_id] = {
        "session": session,
        "role": body.role,
        "skills": body.skills,
        "results": [],
    }
    return SessionResponse(
        id=session_id,
        role=body.role,
        skills=body.skills,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    return SessionResponse(
        id=session_id,
        role=entry["role"],
        skills=entry["skills"],
        current_question=sess.current_question,
        current_category=sess.current_category,
        question_history=sess.question_history,
        results_count=len(entry["results"]),
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    store = _get_store(request)
    if session_id not in store:
        raise HTTPException(404, "Session not found")
    del store[session_id]


@router.post("/{session_id}/questions", response_model=QuestionResponse)
async def generate_question(session_id: str, request: Request):
    start = time.perf_counter()
    logger.info("POST /sessions/%s/questions started", session_id)
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    sess.generate_question(entry["role"], entry["skills"])
    sess.classify_current_question()

    elapsed = time.perf_counter() - start
    threshold_warn = 8000
    if elapsed >= threshold_warn:
        logger.warning(
            "SLOW | POST /sessions/%s/questions took %.2f seconds (>= %.0f s) | role='%s' | skills='%s'",
            session_id, elapsed, threshold_warn, entry["role"], entry["skills"],
        )
    else:
        logger.info(
            "POST /sessions/%s/questions completed in %.2f seconds | role='%s'",
            session_id, elapsed, entry["role"],
        )

    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.get("/{session_id}/current-question", response_model=QuestionResponse)
async def get_current_question(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    if not sess.current_question:
        raise HTTPException(404, "No question generated yet")
    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.post("/{session_id}/answers", response_model=AnswerResponse)
async def submit_answer(session_id: str, request: Request, file: UploadFile = File(...)):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]

    filename = file.filename or "audio.ogg"
    suffix = Path(filename).suffix.lower() or ".ogg"
    audio_bytes = await file.read()

    result = sess.evaluate_answer(audio_bytes, suffix)
    ev_score = result.get("evaluation", {}).get("score")
    print(ev_score)
    if ev_score is None:
        logger.warning("Storing result with score=None, coercing to 0")
        result["evaluation"]["score"] = 0
    entry["results"].append(result)
    ev = result.get("evaluation", {})
    return AnswerResponse(
        question=result["question"],
        category=result.get("category"),
        transcript=result.get("transcript", ""),
        evaluation=EvaluationSchema(
            score=ev.get("score"),
            feedback=ev.get("feedback"),
            status=ev.get("status"),
            message=ev.get("message"),
        ),
    )


@router.get("/{session_id}/answers", response_model=list[AnswerResponse])
async def get_answers(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    results = []
    for r in entry["results"]:
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationSchema(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return results


@router.get("/{session_id}/summary", response_model=SummaryResponse)
async def get_summary(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    summary = sess.finish(entry["results"])
    results = []
    for r in summary.get("results", entry["results"]):
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationSchema(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return SummaryResponse(
        total_questions=summary["total_questions"],
        evaluated=summary["evaluated"],
        final_score=summary["final_score"],
        results=results,
    )
