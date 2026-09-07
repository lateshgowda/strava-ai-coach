"""
AI coaching layer — uses OpenRouter to power the conversational coach.

IMPORTANT: This module ONLY interprets, explains, and coaches.
ALL metrics (fatigue, ACWR, paces, predictions) are computed by the
deterministic analytics engine BEFORE being passed here.
The LLM never computes training numbers — it explains them.
"""
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-14b")

_EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Strava AI Coach",
}


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_EXTRA_HEADERS,
    )


# ---------------------------------------------------------------------------
# One-shot insight
# ---------------------------------------------------------------------------


def generate_run_insight(metrics: Dict[str, Any]) -> str:
    """
    Generate a focused, evidence-based coaching insight for recent training.
    Concise (200–300 words), direct, no fluff.
    """
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured. Please add OPENROUTER_API_KEY to your .env file."

    metrics_text = _format_metrics(metrics)

    prompt = f"""/no_think
You are an experienced endurance coach analysing an athlete's training data.
Be direct, evidence-based, and specific. No motivational fluff.

ATHLETE METRICS:
{metrics_text}

Provide a coaching analysis with exactly these four sections:
1. STRONGEST IMPROVEMENT: What is genuinely improving and why it matters
2. BIGGEST LIMITER: The single most important factor holding performance back
3. KEY RISK: Any injury or overtraining risk to flag (or "None detected")
4. RECOMMENDED FOCUS: One specific, actionable thing to prioritise this week

Rules:
- Reference actual numbers from the data
- 200–300 words total
- If ACWR > 1.3: flag injury risk clearly
- If fatigue > 70: emphasise recovery
- No generic advice — every sentence must be specific to this athlete's data
- No medical advice
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "No insight generated."
    except Exception as exc:
        return f"Unable to generate insight: {exc}"


# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------


def generate_weekly_summary(weekly_data: Dict[str, Any]) -> str:
    """Generate a structured weekly review with next-week recommendations."""
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    metrics_text = _format_metrics(weekly_data)

    prompt = f"""/no_think
You are an endurance coach reviewing an athlete's weekly training.
Be specific, direct, and evidence-based. No generic encouragement.

WEEKLY TRAINING DATA:
{metrics_text}

Structure your response with these headings:
**This Week** — summarise volume, intensity distribution, and consistency
**Assessment** — evaluate training load (sustainable? too much? too little?) referencing the ACWR and fatigue data
**Next Week Focus** — give 2–3 specific, actionable recommendations

Rules:
- Reference actual numbers
- 200–300 words
- If ACWR > 1.3: prioritise recovery and load reduction
- If ACWR < 0.8: suggest a modest 5–10% volume increase
- No medical advice, no generic motivational statements
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "No summary generated."
    except Exception as exc:
        return f"Unable to generate summary: {exc}"


# ---------------------------------------------------------------------------
# Conversational coach
# ---------------------------------------------------------------------------


def chat_with_coach(
    message: str,
    history: List[Dict[str, str]],
    metrics: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Conversational AI coach with full athlete context in every turn.

    The system prompt includes:
    - Current training metrics (fatigue, recovery, ACWR, trends)
    - Readiness score and recommendation
    - Race predictions and fitness trajectory
    - Training state (consistency, monotony, HR efficiency)
    - Athlete profile (if set)

    History is provided to maintain context across turns.
    Persistent memory from prior sessions is injected as older history entries.
    """
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured. Please add OPENROUTER_API_KEY to your .env file."

    metrics_text = _format_metrics(metrics)

    # Build profile section
    profile_section = ""
    if profile:
        profile_lines = []
        if profile.get("age"):
            profile_lines.append(f"Age: {profile['age']}")
        if profile.get("running_experience"):
            profile_lines.append(f"Experience: {profile['running_experience']}")
        if profile.get("primary_goal"):
            profile_lines.append(f"Primary goal: {profile['primary_goal']}")
        if profile.get("target_race_distance"):
            profile_lines.append(f"Target race: {profile['target_race_distance']}")
        if profile.get("injury_history"):
            profile_lines.append(f"Injury history: {profile['injury_history']}")
        if profile.get("preferred_weekly_km"):
            profile_lines.append(f"Preferred weekly km: {profile['preferred_weekly_km']}")
        if profile_lines:
            profile_section = "\nATHLETE PROFILE:\n" + "\n".join(profile_lines)

    system_prompt = f"""/no_think
You are a highly experienced endurance coach and sports scientist specialising in \
marathon, half-marathon, and 10K training. You analyse data like a professional and \
communicate like a trusted training partner — direct, specific, evidence-based.

You have access to this athlete's complete training data below. Use it in every answer.
{profile_section}

CURRENT TRAINING METRICS:
{metrics_text}

CRITICAL — DATA AVAILABILITY RULES:
- The CURRENT TRAINING METRICS block above is the ONLY source of truth. It supersedes anything said in prior conversation history.
- If sleep_last_2_nights appears in the metrics above, sleep data IS available — reference it directly with the exact values shown.
- If resting_hr_bpm appears in sleep_last_2_nights, that is the morning HR from the athlete's wearable.
- NEVER say you don't have data that is present in the metrics block above.

HOW TO RESPOND:
- Always reference the athlete's actual numbers — never give generic advice
- Explain the WHY behind every recommendation (the physiology, not just the what)
- Compare current vs historical trends when relevant
- If fatigue_score > 70 or ACWR > 1.3: flag recovery need prominently
- If readiness_score < 45: redirect hard workout questions to easier alternatives
- For race questions: use the race_predictions data and explain confidence
- For HR drift questions: use hr_efficiency_trend and pace_efficiency_trend data
- For long run fade: use the long_run_pace_fade verdict and data
- For sleep questions: use sleep_last_2_nights data — total_sleep_hours, deep_sleep_hours, rem_sleep_hours
- Keep responses 150–250 words unless a detailed breakdown is requested
- Use a supportive but honest tone — do not sugarcoat problems
- No medical advice. No hallucinated metrics — only reference data you have.
"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            max_tokens=700,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "I couldn't generate a response. Please try again."
    except Exception as exc:
        return f"Unable to reach AI coach: {exc}"


# ---------------------------------------------------------------------------
# FM Plan — per-run review
# ---------------------------------------------------------------------------


def generate_plan_run_review(planned: Dict[str, Any], actual: Dict[str, Any]) -> str:
    """2-3 sentence coaching review for a single run vs the plan."""
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    prompt = f"""/no_think
You are an experienced marathon coach reviewing a training run against the planned session.
Be direct, specific, and constructive. Respond in exactly 3 sentences.

PLANNED:
- Date: {planned.get('plan_date')}
- Session type: {planned.get('session_type')}
- Target distance: {planned.get('distance_km', 0):.1f} km
- Coach notes: {planned.get('details') or 'No target details'}

ACTUAL:
- Date: {actual.get('date')}
- Distance: {actual.get('distance_km', 0):.2f} km
- Avg HR: {actual.get('avg_hr') or 'N/A'} bpm
- Pace: {actual.get('pace') or 'N/A'} /km
- Cadence (SPM): {actual.get('spm') or 'N/A'}

Write exactly 3 sentences:
1. Whether the session was executed as planned (compare distance and any HR/pace targets)
2. One specific observation about execution quality
3. One actionable tip for the next similar session
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Could not generate review."
    except Exception as exc:
        return f"Review generation failed: {exc}"


# ---------------------------------------------------------------------------
# FM Plan — monthly summary
# ---------------------------------------------------------------------------


def generate_plan_monthly_summary(month_data: Dict[str, Any]) -> str:
    """Comprehensive monthly coaching review assessing FM Sep 2027 readiness."""
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    display = month_data.get("display", month_data.get("year_month"))
    month_num = month_data.get("month_num", "?")
    planned_km = month_data.get("planned_km", 0)
    actual_km = month_data.get("actual_km", 0)
    completed = month_data.get("completed", 0)
    missed = month_data.get("missed", 0)
    upcoming = month_data.get("upcoming", 0)
    completion_pct = month_data.get("completion_pct", 0)

    # Build per-run detail
    run_lines = []
    for i, run in enumerate(month_data.get("workouts", []), 1):
        planned_str = f"{run['distance_km']:.1f}km {run['session_type']}"
        if run["status"] == "completed" and run.get("actual"):
            a = run["actual"]
            pace_str = f"@ {a['pace']}/km" if a.get("pace") else ""
            hr_str = f"HR {a['avg_hr']}" if a.get("avg_hr") else ""
            run_lines.append(
                f"  Run {i} ({run['plan_date']}): DONE — Planned {planned_str} | "
                f"Actual {a['distance_km']:.1f}km {pace_str} {hr_str}".strip()
            )
        elif run["status"] == "missed":
            run_lines.append(f"  Run {i} ({run['plan_date']}): MISSED — Planned {planned_str}")
        else:
            run_lines.append(f"  Run {i} ({run['plan_date']}): UPCOMING — Planned {planned_str}")

    prompt = f"""/no_think
You are an experienced marathon coach writing a monthly training review for an athlete \
targeting a sub-5-hour full marathon in September 2027. This is training month {month_num} of ~13. \
Target race pace: 6:30–6:40/km. The plan has 3 runs per week (Tue/Thu/Sat).

MONTH: {display}
Planned volume: {planned_km:.1f} km | Actual volume: {actual_km:.1f} km | Completion: {completion_pct}%
Runs completed: {completed} | Missed: {missed} | Upcoming: {upcoming}

ALL RUNS THIS MONTH:
{chr(10).join(run_lines)}

Write a comprehensive 5–6 sentence monthly coaching review covering:
1. Volume execution — how closely did actual match planned km?
2. Consistency — how many sessions were completed vs missed, and what pattern do you notice?
3. Quality observations — any standout pace, HR, or cadence trends from the completed runs?
4. FM readiness assessment — based on month {month_num} of 13, is the athlete on track for Sep 2027?
5. Biggest win this month and biggest area to improve
6. One specific focus and measurable target for next month

Be direct, specific, and reference actual numbers. No generic statements.
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Could not generate monthly summary."
    except Exception as exc:
        return f"Monthly summary generation failed: {exc}"


# ---------------------------------------------------------------------------
# FM Plan — weekly summary
# ---------------------------------------------------------------------------


def generate_plan_weekly_summary(iso_week: str, week_runs: List[Dict[str, Any]]) -> str:
    """4-sentence weekly coaching summary for FM plan progress."""
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured."

    runs_lines = []
    for i, run in enumerate(week_runs, 1):
        planned_str = f"{run['distance_km']:.1f}km {run['session_type']}"
        if run["status"] == "completed" and run.get("actual"):
            a = run["actual"]
            pace_str = f"@ {a['pace']}/km" if a.get("pace") else ""
            hr_str = f"HR {a['avg_hr']}" if a.get("avg_hr") else ""
            runs_lines.append(
                f"Run {i}: DONE — Planned {planned_str} | Actual {a['distance_km']:.1f}km {pace_str} {hr_str}".strip()
            )
        elif run["status"] == "missed":
            runs_lines.append(f"Run {i}: MISSED — Planned {planned_str}")
        else:
            runs_lines.append(f"Run {i}: UPCOMING — Planned {planned_str}")

    prompt = f"""/no_think
You are an experienced marathon coach writing a weekly review for an athlete targeting a \
full marathon in September 2027. Target finish pace: 6:30–6:40/km.

WEEK: {iso_week}
RUNS:
{chr(10).join(runs_lines)}

Write exactly 4 sentences covering:
1. Overall execution of the week (completed, partial, or missed)
2. What went well — be specific, reference actual numbers
3. What needs improvement — be specific and honest
4. One concrete focus for next week
"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Could not generate summary."
    except Exception as exc:
        return f"Summary generation failed: {exc}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_metrics(metrics: Dict[str, Any], indent: int = 0) -> str:
    """Recursively format a metrics dict into a readable plain-text block."""
    lines = []
    prefix = "  " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_metrics(value, indent + 1))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Render each item as an indented block (e.g. recent_activities)
                lines.append(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    lines.append(f"{prefix}  [{i + 1}] {', '.join(f'{k}={v}' for k, v in item.items() if v is not None)}")
            else:
                lines.append(f"{prefix}{key}: [{len(value)} items]")
        elif value is None:
            lines.append(f"{prefix}{key}: N/A")
        elif isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.2f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)
