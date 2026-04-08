#!/usr/bin/env python3
"""Baseline inference agent for SignAdapt.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN, and OPENAI_API_KEY from env vars.
Uses the OpenAI client to select tutoring actions step-by-step.
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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

MAX_SAFETY_STEPS = 15

SYSTEM_PROMPT = (
    "You are a sign-language tutoring planner. "
    "Given the current observation, choose the single best next action.\n"
    "Reply with ONLY a JSON object: "
    '{"action_type": "<one of the allowed actions>", "rationale": "<brief reason>", "payload": {}}\n'
    "Allowed actions: " + ", ".join(ALLOWED_ACTIONS)
)


def _build_user_prompt(obs: dict) -> str:
    parts = [
        f"Task: {obs.get('task_id')} ({obs.get('difficulty')})",
        f"Target sign: {obs.get('lesson_goal', {}).get('target_sign', '?')}",
        f"Error patterns: {json.dumps(obs.get('error_patterns', []))}",
        f"Support needs: {obs.get('support_needs', [])}",
        f"Coverage: {json.dumps(obs.get('coverage', {}))}",
        f"Completed requirements: {obs.get('completed_requirements', [])}",
        f"Tutoring plan so far: {obs.get('tutoring_plan', [])}",
        f"Remaining steps: {obs.get('remaining_steps', 0)}",
        f"Step count: {obs.get('step_count', 0)}",
    ]
    if obs.get("last_action_result"):
        parts.append(f"Last result: {obs['last_action_result']}")
    return "\n".join(parts)


def _fallback_policy(obs: dict) -> dict:
    """Heuristic fallback when LLM output is invalid."""
    coverage = obs.get("coverage", {})
    errors = {e.get("error_type") for e in obs.get("error_patterns", [])}
    needs = set(obs.get("support_needs", []))
    plan = obs.get("tutoring_plan", [])
    remaining = obs.get("remaining_steps", 1)
    difficulty = obs.get("difficulty", "easy")

    if remaining <= 1:
        return {"action_type": "finalize_plan", "rationale": "Budget exhausted", "payload": {}}

    prereqs = obs.get("lesson_goal", {}).get("prerequisite_signs", [])
    if prereqs and not coverage.get("has_prerequisite"):
        return {"action_type": "select_prerequisite_sign", "rationale": "Review prerequisite first", "payload": {}}

    if ("movement" in errors or "timing" in errors) and not coverage.get("has_timing_support"):
        return {"action_type": "slow_motion_demo", "rationale": "Address movement/timing with slow demo", "payload": {}}

    if "location" in errors and not coverage.get("has_visual_cue"):
        return {"action_type": "add_location_cue", "rationale": "Address location error", "payload": {}}

    if ("movement" in errors) and not coverage.get("has_movement_hint"):
        return {"action_type": "add_movement_hint", "rationale": "Provide movement guidance", "payload": {}}

    if not coverage.get("has_drill"):
        return {"action_type": "generate_micro_drill", "rationale": "Practice drill needed", "payload": {}}

    if "attention_support" in needs and not coverage.get("has_feedback_style"):
        return {"action_type": "choose_feedback_style", "rationale": "Support attention needs", "payload": {}}

    if not coverage.get("has_assessment"):
        return {"action_type": "quick_assessment", "rationale": "Check learner progress", "payload": {}}

    if difficulty == "hard" and not coverage.get("has_revision"):
        return {"action_type": "revision_loop", "rationale": "Reinforce through revision", "payload": {}}

    return {"action_type": "finalize_plan", "rationale": "All steps complete", "payload": {}}


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


def run_episode(task_id: str) -> None:
    print(f"[START] Running episode for task={task_id}")

    # Reset
    resp = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    episode_id = data["episode_id"]
    obs = data["observation"]
    done = data["done"]

    # Setup OpenAI client
    api_key = OPENAI_API_KEY or HF_TOKEN or "dummy"
    base_url = None
    if HF_TOKEN and not OPENAI_API_KEY:
        base_url = API_BASE_URL if API_BASE_URL.startswith("http") else None

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    step_num = 0
    while not done and step_num < MAX_SAFETY_STEPS:
        step_num += 1

        # Try LLM
        action_dict = None
        try:
            user_prompt = _build_user_prompt(obs)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            raw = completion.choices[0].message.content or ""
            action_dict = _parse_action(raw)
        except Exception as exc:
            print(f"[STEP {step_num}] LLM call failed: {exc}; using fallback")

        if action_dict is None:
            action_dict = _fallback_policy(obs)
            print(f"[STEP {step_num}] Fallback action: {action_dict['action_type']}")
        else:
            print(f"[STEP {step_num}] LLM action: {action_dict['action_type']} — {action_dict.get('rationale', '')}")

        # Post step
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"episode_id": episode_id, "action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})

        print(f"[STEP {step_num}] reward={reward:.4f}  done={done}")

        if done and "final_grade" in info:
            grade = info["final_grade"]
            print(f"[END] Final score={grade['total_score']:.4f}  passed={grade['passed']}")
            print(f"[END] Reasoning: {grade['reasoning']}")
            if grade.get("missing_requirements"):
                print(f"[END] Missing: {grade['missing_requirements']}")
            return

    print("[END] Episode complete (max safety steps reached)")


def main() -> None:
    task_ids = ["easy_remediate_handshape", "medium_fix_movement_with_scaffold", "hard_adaptive_multi_error_plan"]
    target = sys.argv[1] if len(sys.argv) > 1 else task_ids[0]
    if target == "all":
        for tid in task_ids:
            run_episode(tid)
            print()
    else:
        run_episode(target)


if __name__ == "__main__":
    main()
