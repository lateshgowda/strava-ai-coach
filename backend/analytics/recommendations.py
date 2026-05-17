"""
Adaptive weekly training recommendation engine.

Generates a concrete, personalised weekly training plan based on current
training state, athlete profile, and sports science principles.

All output is deterministic Python — the LLM explains and contextualises,
but does NOT generate the numbers.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def generate_weekly_plan(
    # Load metrics
    acwr: float,
    fatigue_score: float,
    recovery_score: float,
    readiness_score: float,
    # Recent training
    this_week_km: float,
    avg_weekly_km_last4w: float,
    consistency_score: float,
    consecutive_training_days: int,
    # Athlete profile (optional)
    preferred_weekly_km: Optional[float] = None,
    primary_goal: Optional[str] = None,
    target_race_date_days: Optional[int] = None,
    weekly_training_hours: Optional[float] = None,
    running_experience: Optional[str] = None,
    max_hr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns a structured weekly plan dict.

    The plan adjusts dynamically based on fatigue, ACWR, consistency,
    athlete experience, and proximity to a target race.
    """
    # ── Base weekly km ────────────────────────────────────────────────────
    base_km = max(avg_weekly_km_last4w, 10.0)

    # Determine upper ceiling
    if preferred_weekly_km and preferred_weekly_km > base_km:
        ceiling_km = preferred_weekly_km
    else:
        ceiling_km = base_km * 1.20  # max 20% increase from current

    # Experience-based floor multiplier
    exp_floor = {"beginner": 0.70, "intermediate": 0.80, "advanced": 0.85}.get(
        running_experience or "intermediate", 0.80
    )

    # ── Intensity budget ──────────────────────────────────────────────────
    if acwr > 1.3 or fatigue_score > 70 or readiness_score < 35:
        km_multiplier = 0.80
        intensity_budget = "low"
        phase = "recovery_week"
    elif acwr > 1.1 or fatigue_score > 55 or readiness_score < 52:
        km_multiplier = 0.90
        intensity_budget = "moderate"
        phase = "consolidation"
    elif acwr < 0.8 and consistency_score > 60 and readiness_score >= 65:
        km_multiplier = 1.10
        intensity_budget = "high"
        phase = "build"
    else:
        km_multiplier = 1.00
        intensity_budget = "moderate_high"
        phase = "maintenance"

    # Race taper override
    if target_race_date_days is not None:
        if target_race_date_days <= 7:
            km_multiplier = 0.50
            intensity_budget = "very_low"
            phase = "race_week"
        elif target_race_date_days <= 14:
            km_multiplier = 0.60
            intensity_budget = "low"
            phase = "taper"
        elif target_race_date_days <= 21:
            km_multiplier = 0.70
            intensity_budget = "moderate"
            phase = "pre_taper"
        elif target_race_date_days <= 42:
            km_multiplier = min(km_multiplier, 1.05)
            phase = "peak_build" if phase == "build" else phase

    recommended_km = round(
        max(base_km * exp_floor, min(base_km * km_multiplier, ceiling_km)), 1
    )

    # ── Day structure ─────────────────────────────────────────────────────
    if intensity_budget in ("very_low", "low"):
        total_days = max(3, min(4, round(recommended_km / max(base_km / 5, 1))))
        quality_days = 0
    elif intensity_budget == "moderate":
        total_days = max(4, min(5, round(recommended_km / max(base_km / 5, 1))))
        quality_days = 1
    else:
        total_days = max(4, min(6, round(recommended_km / max(base_km / 5, 1))))
        quality_days = 2 if intensity_budget == "high" else 1

    easy_days = total_days - quality_days

    # Long run: 28–33% of weekly volume
    long_run_km = round(min(recommended_km * 0.30, 28.0), 1)
    long_run_km = max(8.0, long_run_km)

    if phase == "race_week":
        long_run_km = min(long_run_km, 8.0)
    elif phase == "taper":
        long_run_km = round(long_run_km * 0.6, 1)

    # ── Goal-specific workout suggestions ─────────────────────────────────
    goal = (primary_goal or "base_building").lower()

    if goal in ("hm", "half_marathon"):
        interval_desc = "6 × 1 km at current 10K effort (90 sec recovery jog)"
        threshold_desc = "25–30 min tempo at half-marathon goal pace"
        goal_label = "Half Marathon"
    elif goal == "10k":
        interval_desc = "8 × 600 m at 5K effort (90 sec recovery jog)"
        threshold_desc = "20 min tempo at 10K goal pace"
        goal_label = "10K"
    elif goal == "marathon":
        interval_desc = "4 × 2 km at HM race pace (2 min recovery jog)"
        threshold_desc = "35 min at marathon goal pace (with 10 min easy warm-up)"
        goal_label = "Marathon"
    elif goal == "weight_loss":
        interval_desc = "4 × 3 min moderate-hard effort (2 min easy between)"
        threshold_desc = "30 min steady run at conversational pace (Zone 2)"
        goal_label = "Weight Loss / Fitness"
    else:  # base_building / fitness
        interval_desc = "5 × 1 km at comfortably hard effort (2 min recovery jog)"
        threshold_desc = "20 min tempo run at lactate threshold feel"
        goal_label = "Base Building"

    # Suppress hard workouts if intensity budget is low
    interval_workout: Optional[str] = interval_desc if quality_days >= 1 else None
    threshold_workout: Optional[str] = threshold_desc if quality_days >= 2 else None

    # ── Long run guidance ─────────────────────────────────────────────────
    if phase == "race_week":
        long_run_note = "Race week — no long run. Short shake-out jog (20–30 min) at most."
    elif phase == "taper":
        long_run_note = (
            "Taper phase — reduce long run by 40%. Focus on maintaining pace, not distance. "
            "Race-pace strides at the end of your last long run."
        )
    elif phase == "peak_build":
        long_run_note = (
            "Peak build week — this is your longest long run of the cycle. "
            "Target even or negative splits. Consider fuelling every 45 min if > 90 min."
        )
    elif phase == "recovery_week":
        long_run_note = "Recovery week — shorten long run by 30%. Easy effort, no HR pressure."
    else:
        long_run_note = (
            "Build gradually — aim for even splits. If HR exceeds 80% max, slow down. "
            "Fuel with 30–60 g carbs/hr for runs over 75 minutes."
        )

    # ── Recovery recommendations ─────────────────────────────────────────
    rest_days = 7 - total_days
    if recovery_score < 50 or fatigue_score > 65:
        recovery_note = (
            f"Include {max(2, rest_days)} full rest days this week. "
            "Prioritise 8+ hrs sleep, 1.6 g protein/kg body weight, and adequate hydration. "
            "Consider foam rolling and cold-water immersion post hard sessions."
        )
    elif consecutive_training_days >= 4:
        recovery_note = (
            "Schedule a rest day in the next 1–2 days. "
            "Active recovery (easy walking, yoga, swimming) is fine and recommended."
        )
    else:
        recovery_note = (
            f"{rest_days} rest day(s) this week. "
            "One full rest day mid-week preserves training quality on hard days."
        )

    # ── Pacing guidance ───────────────────────────────────────────────────
    easy_pace_note = "Easy runs: 70–75% max HR. You should be able to hold a full conversation."
    if max_hr:
        easy_hr_low = round(max_hr * 0.70)
        easy_hr_high = round(max_hr * 0.76)
        easy_pace_note = f"Easy runs: {easy_hr_low}–{easy_hr_high} bpm (70–76% of {max_hr} bpm max HR)."

    return {
        "phase": phase,
        "goal_label": goal_label,
        "recommended_weekly_km": recommended_km,
        "total_training_days": total_days,
        "easy_days": easy_days,
        "quality_days": quality_days,
        "rest_days": rest_days,
        "long_run_km": long_run_km,
        "long_run_note": long_run_note,
        "intensity_budget": intensity_budget,
        "interval_workout": interval_workout,
        "threshold_workout": threshold_workout,
        "easy_pace_note": easy_pace_note,
        "recovery_note": recovery_note,
        "km_vs_last_week_pct": round(
            (recommended_km - this_week_km) / max(this_week_km, 1) * 100, 1
        ),
    }
