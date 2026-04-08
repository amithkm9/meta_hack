# SignAdapt — Adaptive Sign-Language Tutoring Environment

An **OpenEnv-style** environment where an AI agent learns to plan adaptive sign-language teaching interventions for deaf or hard-of-hearing learners.

## Problem Statement

Teaching sign language is inherently sequential and adaptive. A good tutor observes a learner's specific errors (handshape, movement, location, timing, orientation), considers their cognitive state and support needs, and selects the right interventions in the right order. SignAdapt models this decision process as a structured planning environment that an AI agent can interact with through `reset()`, `step(action)`, and `state()`.

This is **not** a sign-recognition model or a frontend app. It is a **pedagogical planning benchmark** where every state, action, reward, and grade is structured, deterministic, and machine-gradable.

## Why This Environment Is Real-World

- Based on authentic sign-language learning difficulties (handshape errors, movement arc issues, spatial location mistakes, timing problems, orientation errors).
- Reflects real tutoring strategies: slowed demonstrations, prerequisite review, location cues, micro-drills, revision loops, feedback style selection.
- Models adaptive pedagogy: a single intervention rarely fixes complex errors — sequencing and ordering matter.
- Simulates learner cognitive state (confidence, comprehension, frustration, engagement, error reduction) that responds dynamically to tutoring actions.
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
| `learner_state` | Simulated cognitive state (confidence, comprehension, frustration, engagement, error_reduction) |
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

| Task | Difficulty | Target Sign | Max Steps | Error Types |
|---|---|---|---|---|
| `easy_remediate_handshape` | Easy | HELLO | 6 | handshape |
| `easy_fix_orientation` | Easy | PLEASE | 5 | orientation |
| `medium_fix_movement_with_scaffold` | Medium | THANK-YOU | 8 | movement, timing |
| `hard_adaptive_multi_error_plan` | Hard | FAMILY | 10 | movement, location, handshape |
| `hard_complex_sentence_tutoring` | Hard | WANT | 10 | handshape, movement, timing, location |

## Reward & Grading

**Intermediate reward** (per step) is computed from 6 weighted components:

| Component | Weight | Description |
|---|---|---|
| Intervention relevance | 0.20 | Does the action address the learner's actual errors? |
| Pedagogical sequence | 0.20 | Does the action order match ideal pedagogy? |
| Learner need alignment | 0.15 | Does the action support the learner's specific needs? |
| Task completeness | 0.20 | How many required coverage flags are met? |
| Efficiency | 0.10 | Are actions non-redundant? |
| Learner state quality | 0.15 | Is the simulated learner state improving? |

**Final grade** is computed deterministically when `finalize_plan` is called or steps are exhausted. Same 6 components, normalized to `[0.0, 1.0]`. Pass threshold: >= 0.70.

The grader also enforces **ordering constraints** (e.g., prerequisite before drill, assessment before revision) and penalizes violations.

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
pytest tests/ -v
```

## Run Inference

```bash
export ENV_BASE_URL=http://localhost:7860
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
export HF_TOKEN=your-key-here

python inference.py
```

The baseline agent uses a heuristic fallback policy when LLM output is unavailable or invalid.

Inference output uses structured `[START]`, `[STEP]`, `[END]` format:

```
[START] task=easy_remediate_handshape env=signadapt model=gpt-3.5-turbo
[STEP] step=1 action={"action_type":"add_location_cue",...} reward=0.45 done=false error=null
[STEP] step=2 action={"action_type":"quick_assessment",...} reward=0.52 done=false error=null
[STEP] step=3 action={"action_type":"finalize_plan",...} reward=0.38 done=true error=null
[END] success=true steps=3 score=0.650 rewards=0.45,0.52,0.38
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (returns `{"status": "healthy"}`) |
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
