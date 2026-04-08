# SignAdapt — Adaptive Sign-Language Tutoring Environment

An **OpenEnv-style** environment where an AI agent learns to plan adaptive sign-language teaching interventions for deaf or hard-of-hearing learners.

## Problem Statement

Teaching sign language is inherently sequential and adaptive. A good tutor observes a learner's specific errors (handshape, movement, location, timing), considers their support needs, and selects the right interventions in the right order. SignAdapt models this decision process as a structured planning environment that an AI agent can interact with through `reset()`, `step(action)`, and `state()`.

This is **not** a sign-recognition model or a frontend app. It is a **pedagogical planning benchmark** where every state, action, and grade is structured and machine-gradable.

## Why This Environment Is Real-World

- Based on authentic sign-language learning difficulties (handshape errors, movement arc issues, spatial location mistakes, timing problems).
- Reflects real tutoring strategies: slowed demonstrations, prerequisite review, location cues, micro-drills, revision loops.
- Models adaptive pedagogy: a single intervention rarely fixes complex errors — sequencing matters.
- Targets accessibility: improving sign-language education for deaf and hard-of-hearing communities.

## Observation Space

Each observation includes:

| Field | Description |
|---|---|
| `task_id` | Current task identifier |
| `difficulty` | easy / medium / hard |
| `learner` | Age band, proficiency, support needs |
| `lesson_goal` | Target sign, description, prerequisites |
| `error_patterns` | List of errors with type and severity |
| `support_needs` | Learner's support requirements |
| `tutoring_plan` | Actions taken so far |
| `completed_requirements` | Which task requirements have been met |
| `coverage` | Structured flags (has_prerequisite, has_visual_cue, etc.) |
| `remaining_steps` | Steps left in the budget |
| `allowed_actions` | Available action types |
| `last_action_result` | Feedback from the previous step |

## Action Space

Fixed enum of tutoring interventions:

- `select_prerequisite_sign` — Review a prerequisite sign before the target
- `slow_motion_demo` — Provide a slowed demonstration for timing/movement
- `add_location_cue` — Add a spatial location cue
- `add_movement_hint` — Provide movement guidance
- `choose_feedback_style` — Select feedback modality (visual, haptic, verbal, modeling)
- `generate_micro_drill` — Create a focused practice drill
- `quick_assessment` — Check learner progress
- `revision_loop` — Reinforce corrections through revision
- `finalize_plan` — Complete the tutoring plan and trigger grading

Each action is submitted as JSON with `action_type`, `rationale`, and `payload`.

## Task Set

| Task | Difficulty | Target Sign | Max Steps |
|---|---|---|---|
| `easy_remediate_handshape` | Easy | HELLO | 6 |
| `medium_fix_movement_with_scaffold` | Medium | THANK-YOU | 8 |
| `hard_adaptive_multi_error_plan` | Hard | FAMILY | 10 |

## Reward Logic

**Intermediate reward** (per step) is computed from weighted components:

- **Intervention relevance** (0.25) — Does the action address the learner's actual errors?
- **Pedagogical sequence** (0.25) — Does the action order match ideal pedagogy?
- **Learner need alignment** (0.20) — Does the action support the learner's specific needs?
- **Task completeness** (0.20) — How many required coverage flags are met?
- **Efficiency** (0.10) — Are actions non-redundant?

All rewards are normalized to `[0.0, 1.0]`.

## Grader Logic

**Final grade** is computed deterministically when `finalize_plan` is called or steps are exhausted:

- Same five weighted components as above, computed over the full episode.
- Uses longest common subsequence for sequence scoring.
- **Pass threshold**: ≥ 0.70
- Returns structured `GradeReport` with `total_score`, `sub_scores`, `passed`, `reasoning`, and `missing_requirements`.

## Local Setup

```bash
cd signadapt
pip install -r requirements.txt
```

## Run the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Run Tests

```bash
cd signadapt
pytest tests/ -v
```

## Run Inference

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-3.5-turbo
export OPENAI_API_KEY=your-key-here

python inference.py easy_remediate_handshape
python inference.py all
```

The baseline agent uses a heuristic fallback policy when LLM output is invalid.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit an action |
| GET | `/state` | Get current environment state |

## Docker

```bash
docker build -t signadapt .
docker run -p 7860:7860 signadapt
```

Compatible with Hugging Face Spaces Docker deployment.

## Project Structure

```
signadapt/
├── README.md
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── inference.py
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── tasks.py
│   ├── reward.py
│   ├── grader.py
│   ├── env.py
│   ├── routes.py
│   └── sample_data/
│       └── tasks.json
└── tests/
    ├── __init__.py
    ├── test_env.py
    ├── test_api.py
    └── test_grader.py
```
