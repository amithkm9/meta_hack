#!/usr/bin/env python3
"""Baseline inference agent for SignAdapt.

Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
Uses the OpenAI client for LLM calls.
Emits structured stdout logs in [START], [STEP], [END] format.
"""
from __future__ import annotations

import json
import os
import sys

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "signadapt"
REQUEST_TIMEOUT = 30

ALLOWED_ACTIONS = [
    "select_prerequisite_sign",
    "slow_motion_demo",
    "add_location_cue",
    "add_movement_hint",
    "choose_feedback_style",
    "generate_micro_drill",
    "quick_assessment",
    "revision_loop",
    "finalize_plan",
]

SYSTEM_PROMPT = (
    "You are a sign-language tutoring planner for deaf/hard-of-hearing learners. "
    "Given the current observation, choose the single best next tutoring action.\n"
    "Reply with ONLY a minified JSON object with keys: action_type, rationale, payload.\n"
    "action_type must be one of: " + ", ".join(ALLOWED_ACTIONS) + "\n"
    "payload should be an empty object {}.\n"
    "Be deterministic and follow good pedagogical sequencing."
)


def _build_user_prompt(obs: dict) -> str:
    coverage = obs.get("coverage", {})
    learner_state = obs.get("learner_state", {})
    parts = [
        f"Task: {obs.get('task_id')} (difficulty={obs.get('difficulty')})",
        f"Target sign: {obs.get('lesson_goal', {}).get('target_sign', '?')}",
        f"Prerequisites: {obs.get('lesson_goal', {}).get('prerequisite_signs', [])}",
        f"Errors: {json.dumps(obs.get('error_patterns', []), separators=(',',':'))}",
        f"Support needs: {obs.get('support_needs', [])}",
        f"Coverage: {json.dumps(coverage, separators=(',',':'))}",
        f"Learner state: {json.dumps(learner_state, separators=(',',':'))}",
        f"Completed: {obs.get('completed_requirements', [])}",
        f"Plan so far: {[p.split(' — ')[0] if ' — ' in p else p for p in obs.get('tutoring_plan', [])]}",
        f"Remaining steps: {obs.get('remaining_steps', 0)}",
        f"Step: {obs.get('step_count', 0)}",
    ]
    if obs.get("last_action_result"):
        parts.append(f"Last result: {obs['last_action_result']}")
    return "\n".join(parts)


def _fallback_policy(obs: dict) -> dict:
    """Heuristic fallback when LLM output is invalid."""
    coverage = obs.get("coverage", {})
    errors = {e.get("error_type") for e in obs.get("error_patterns", [])}
    needs = set(obs.get("support_needs", []))
    remaining = obs.get("remaining_steps", 1)
    difficulty = obs.get("difficulty", "easy")

    if remaining <= 1:
        return {"action_type": "finalize_plan", "rationale": "Budget exhausted", "payload": {}}

    prereqs = obs.get("lesson_goal", {}).get("prerequisite_signs", [])
    if prereqs and not coverage.get("has_prerequisite"):
        return {"action_type": "select_prerequisite_sign", "rationale": "Review prerequisites first", "payload": {}}

    if ("movement" in errors or "timing" in errors) and not coverage.get("has_timing_support"):
        return {"action_type": "slow_motion_demo", "rationale": "Address movement/timing with slow demo", "payload": {}}

    if "location" in errors and not coverage.get("has_visual_cue"):
        return {"action_type": "add_location_cue", "rationale": "Address location error with spatial cue", "payload": {}}

    if "movement" in errors and not coverage.get("has_movement_hint"):
        return {"action_type": "add_movement_hint", "rationale": "Provide movement guidance", "payload": {}}

    if "attention_support" in needs and not coverage.get("has_feedback_style"):
        return {"action_type": "choose_feedback_style", "rationale": "Support attention needs", "payload": {}}

    if not coverage.get("has_drill"):
        return {"action_type": "generate_micro_drill", "rationale": "Practice drill needed", "payload": {}}

    if not coverage.get("has_assessment"):
        return {"action_type": "quick_assessment", "rationale": "Check learner progress", "payload": {}}

    if difficulty == "hard" and not coverage.get("has_revision"):
        return {"action_type": "revision_loop", "rationale": "Reinforce through revision", "payload": {}}

    return {"action_type": "finalize_plan", "rationale": "All requirements covered", "payload": {}}


def _parse_action(text: str) -> dict | None:
    """Try to extract a JSON action from LLM text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if "action_type" not in obj:
        return None
    if obj["action_type"] not in ALLOWED_ACTIONS:
        return None
    obj.setdefault("rationale", "")
    obj.setdefault("payload", {})
    return obj


def _format_action(action_dict: dict) -> str:
    return json.dumps(action_dict, separators=(",", ":"))


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(
    session: requests.Session,
    client: OpenAI | None,
    base_url: str,
    model_name: str,
    task_id: str,
) -> dict:
    rewards: list[float] = []
    step_count = 0
    done = False
    success = False
    score = 0.0

    _log_start(task_id, model_name)

    try:
        resp = session.post(
            f"{base_url}/reset",
            json={"task_id": task_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        episode_id = data["episode_id"]
        obs = data["observation"]

        while not done:
            action_dict = None

            if client is not None:
                try:
                    user_prompt = _build_user_prompt(obs)
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    raw = completion.choices[0].message.content or ""
                    action_dict = _parse_action(raw)
                except Exception:
                    pass

            if action_dict is None:
                action_dict = _fallback_policy(obs)

            action_str = _format_action(action_dict)

            try:
                step_resp = session.post(
                    f"{base_url}/step",
                    json={"episode_id": episode_id, "action": action_dict},
                    timeout=REQUEST_TIMEOUT,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                obs = result["observation"]
                rewards.append(reward)
                step_count += 1
                _log_step(step_count, action_str, reward, done, None)
            except Exception as exc:
                _log_step(step_count + 1, action_str, 0.0, False, str(exc))
                break

        if done and "info" in result and "final_grade" in result.get("info", {}):
            score = float(result["info"]["final_grade"]["total_score"])
        elif rewards:
            score = round(sum(rewards) / len(rewards), 3)
        score = max(0.0, min(1.0, score))
        success = done and score >= 0.1

        return {
            "task_id": task_id,
            "steps": step_count,
            "total_reward": round(sum(rewards), 3),
            "score": score,
            "success": success,
        }
    except Exception as exc:
        score = 0.0
        return {
            "task_id": task_id,
            "steps": step_count,
            "score": score,
            "success": False,
            "error": str(exc),
        }
    finally:
        _log_end(success=success, steps=step_count, score=score, rewards=rewards)


def main() -> int:
    base_url = ENV_BASE_URL.rstrip("/")
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY") or ""

    session = requests.Session()

    client: OpenAI | None = None
    if api_key and API_BASE_URL:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        except Exception:
            client = None
    elif api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception:
            client = None

    task_ids = _discover_tasks(session, base_url)

    for task_id in task_ids:
        run_task(session, client, base_url, MODEL_NAME, task_id)

    return 0


def _discover_tasks(session: requests.Session, base_url: str) -> list[str]:
    """Fetch task list from the running environment."""
    try:
        resp = session.get(f"{base_url}/tasks", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        tasks = resp.json()
        return [t["id"] for t in tasks]
    except Exception:
        return [
            "easy_remediate_handshape",
            "easy_fix_orientation",
            "medium_fix_movement_with_scaffold",
            "hard_adaptive_multi_error_plan",
            "hard_complex_sentence_tutoring",
        ]


if __name__ == "__main__":
    raise SystemExit(main())
