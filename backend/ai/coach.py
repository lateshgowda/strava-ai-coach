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

HOW TO RESPOND:
- Always reference the athlete's actual numbers — never give generic advice
- Explain the WHY behind every recommendation (the physiology, not just the what)
- Compare current vs historical trends when relevant
- If fatigue_score > 70 or ACWR > 1.3: flag recovery need prominently
- If readiness_score < 45: redirect hard workout questions to easier alternatives
- For race questions: use the race_predictions data and explain confidence
- For HR drift questions: use hr_efficiency_trend and pace_efficiency_trend data
- For long run fade: use the long_run_pace_fade verdict and data
- Keep responses 150–250 words unless a detailed breakdown is requested
- Use a supportive but honest tone — do not sugarcoat problems
- No medical advice. No hallucinated metrics — only reference data you have.
- Do not repeat information the athlete already knows unless they ask
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
