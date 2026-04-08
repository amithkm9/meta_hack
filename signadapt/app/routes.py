from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.env import SignAdaptEnv
from app.models import (
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResult,
    TaskSummary,
)
from app.tasks import list_tasks

router = APIRouter()

_env = SignAdaptEnv()


@router.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@router.get("/tasks", response_model=list[TaskSummary])
def get_tasks() -> list[TaskSummary]:
    return [
        TaskSummary(
            id=t.id,
            difficulty=t.difficulty,
            target_sign=t.lesson_goal.target_sign,
            description=t.lesson_goal.description,
            max_steps=t.constraints.max_steps,
        )
        for t in list_tasks()
    ]


@router.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest) -> ResetResponse:
    try:
        return _env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/step", response_model=StepResult)
def step_env(req: StepRequest) -> StepResult:
    if req.episode_id != _env._episode_id:
        raise HTTPException(status_code=400, detail="Episode ID mismatch.")
    try:
        return _env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
