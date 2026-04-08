from __future__ import annotations

import json
from pathlib import Path

from app.models import TaskSpec

_TASKS_FILE = Path(__file__).parent / "sample_data" / "tasks.json"
_CACHE: list[TaskSpec] | None = None


def _load() -> list[TaskSpec]:
    global _CACHE
    if _CACHE is None:
        raw = json.loads(_TASKS_FILE.read_text())
        _CACHE = [TaskSpec(**t) for t in raw]
    return _CACHE


def reload_tasks() -> list[TaskSpec]:
    """Force-reload tasks from disk (useful for testing)."""
    global _CACHE
    _CACHE = None
    return _load()


def list_tasks() -> list[TaskSpec]:
    return _load()


def get_task(task_id: str) -> TaskSpec:
    for t in _load():
        if t.id == task_id:
            return t.model_copy(deep=True)
    raise ValueError(f"Unknown task_id: {task_id}")


def sample_task() -> TaskSpec:
    return _load()[0].model_copy(deep=True)
