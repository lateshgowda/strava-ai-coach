"""
Daily readiness engine.

Produces a composite 0–100 readiness score and a tiered recommendation
based purely on deterministic training load metrics.  No LLM involved.

The score is inspired by:
- Banister CTL/ATL model
- Recovery research (Kellmann & Kallus, 2001)
- ACWR injury-risk model (Gabbett, 2016)
"""
from __future__ import annotations
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Recommendation tiers
# ---------------------------------------------------------------------------

TIER_HARD = "hard_workout_ok"
TIER_MODERATE = "moderate_ok"
TIER_EASY = "easy_only"
TIER_RECOVERY = "recovery_day"
TIER_REST = "full_rest"

TIER_LABELS: Dict[str, str] = {
    TIER_HARD: "✅ Hard Workout OK",
    TIER_MODERATE: "🟢 Moderate Training OK",
    TIER_EASY: "🟡 Easy Running Only",
    TIER_RECOVERY: "🟠 Recovery Day",
    TIER_REST: "🛑 Full Rest",
}

TIER_COLORS: Dict[str, str] = {
    TIER_HARD: "#48BB78",
    TIER_MODERATE: "#68D391",
    TIER_EASY: "#ECC94B",
    TIER_RECOVERY: "#ED8936",
    TIER_REST: "#F56565",
}

TIER_DESCRIPTIONS: Dict[str, str] = {
    TIER_HARD: (
        "Fatigue is low, recovery is strong, and workload is well balanced. "
        "You are primed for a quality session. Intervals, tempo, or a long run "
        "are all appropriate today."
    ),
    TIER_MODERATE: (
        "Fatigue is manageable and recovery is decent. Steady aerobic running "
        "at comfortable effort is ideal. Save high-intensity work for when "
        "readiness is higher."
    ),
    TIER_EASY: (
        "Fatigue is elevated or recovery is incomplete. Keep today's effort fully "
        "conversational — target 70–75% max HR. No tempo or interval work. "
        "Even a short easy run still provides aerobic benefit."
    ),
    TIER_RECOVERY: (
        "Significant fatigue has accumulated or workload ratio is high. Active "
        "recovery only: easy walking, yoga, or light stretching. Skip structured "
        "running — adaptation happens during rest, not just training."
    ),
    TIER_REST: (
        "Training load is unsustainably high or recovery is critically low. "
        "A full rest day today will produce better long-term gains than any "
        "workout. Sleep, nutrition, and hydration are your training today."
    ),
}


def compute_readiness(
    fatigue_score: float,
    recovery_score: float,
    acwr: float,
    monotony_score: float,
    consecutive_training_days: int,
    recovery_debt: float,
    overreaching: bool,
    sleep_hours: float | None = None,
    sleep_deep_hours: float | None = None,
    resting_hr: int | None = None,
    baseline_resting_hr: int | None = None,
) -> Dict[str, Any]:
    """
    Compute a 0–100 readiness score and a tiered recommendation.

    Parameters
    ----------
    fatigue_score          : 0–100 (higher = more fatigued)
    recovery_score         : 0–100 (higher = better recovered)
    acwr                   : acute:chronic workload ratio
    monotony_score         : Banister monotony (mean/std daily load over 28d)
    consecutive_training_days : unbroken training streak
    recovery_debt          : (ATL-CTL)/CTL — positive = accumulated debt
    overreaching           : boolean flag from TrainingStateEngine
    sleep_hours            : last night's total sleep in hours (from Apple Health)
    sleep_deep_hours       : deep/slow-wave sleep in hours
    resting_hr             : morning resting HR (bpm)
    baseline_resting_hr    : athlete's typical resting HR (from profile)
    """
    # Base: map recovery score (0-100) → start score (55-100)
    score = 55.0 + recovery_score * 0.45

    # Fatigue penalty: 0 at fatigue ≤ 30, up to -30 at fatigue = 100
    fatigue_penalty = max(0.0, (fatigue_score - 30.0) / 70.0 * 30.0)
    score -= fatigue_penalty

    # ACWR penalty
    if acwr > 1.0:
        acwr_penalty = min(25.0, (acwr - 1.0) / 0.5 * 25.0)
    elif 0 < acwr < 0.6:
        acwr_penalty = 5.0
    else:
        acwr_penalty = 0.0
    score -= acwr_penalty

    # Monotony penalty: 0 below 1.5, up to -12 above 2.5
    if monotony_score > 1.5:
        mono_penalty = min(12.0, (monotony_score - 1.5) * 8.0)
        score -= mono_penalty

    # Consecutive training days penalty: > 5 days → -2 per extra day
    if consecutive_training_days > 5:
        consec_penalty = min(8.0, (consecutive_training_days - 5) * 2.0)
        score -= consec_penalty

    # Recovery debt penalty
    if recovery_debt > 0.2:
        debt_penalty = min(12.0, (recovery_debt - 0.2) * 20.0)
        score -= debt_penalty

    # ------------------------------------------------------------------
    # Sleep penalty (Apple Health data — only applied when available)
    # ------------------------------------------------------------------
    sleep_penalty = 0.0
    deep_sleep_penalty = 0.0
    if sleep_hours is not None:
        if sleep_hours < 5.0:
            sleep_penalty = 20.0
        elif sleep_hours < 6.0:
            sleep_penalty = 13.0
        elif sleep_hours < 7.0:
            sleep_penalty = 6.0
        # ≥ 7h → no penalty
        score -= sleep_penalty

    if sleep_deep_hours is not None:
        if sleep_deep_hours < 0.5:
            deep_sleep_penalty = 10.0
        elif sleep_deep_hours < 0.75:
            deep_sleep_penalty = 6.0
        elif sleep_deep_hours < 1.0:
            deep_sleep_penalty = 3.0
        # ≥ 1h deep sleep → no penalty
        score -= deep_sleep_penalty

    # ------------------------------------------------------------------
    # Resting HR elevation penalty (only when both today's and baseline exist)
    # ------------------------------------------------------------------
    resting_hr_penalty = 0.0
    if resting_hr is not None and baseline_resting_hr is not None and baseline_resting_hr > 0:
        hr_elevation = resting_hr - baseline_resting_hr
        if hr_elevation > 10:
            resting_hr_penalty = 15.0
        elif hr_elevation > 7:
            resting_hr_penalty = 10.0
        elif hr_elevation > 4:
            resting_hr_penalty = 5.0
        elif hr_elevation > 2:
            resting_hr_penalty = 2.0
        score -= resting_hr_penalty

    # Overreaching hard cap
    if overreaching:
        score = min(score, 30.0)

    score = round(max(0.0, min(100.0, score)), 1)

    # Tier assignment
    if overreaching or score < 20:
        tier = TIER_REST
    elif score < 35:
        tier = TIER_RECOVERY
    elif score < 52:
        tier = TIER_EASY
    elif score < 68:
        tier = TIER_MODERATE
    else:
        tier = TIER_HARD

    factors: Dict[str, Any] = {
        "fatigue_penalty": round(fatigue_penalty, 1),
        "acwr_penalty": round(acwr_penalty, 1),
        "consecutive_days": consecutive_training_days,
        "overreaching": overreaching,
        "recovery_debt": recovery_debt,
    }
    if sleep_hours is not None:
        factors["sleep_hours"] = round(sleep_hours, 1)
        factors["sleep_penalty"] = round(sleep_penalty, 1)
    if sleep_deep_hours is not None:
        factors["sleep_deep_hours"] = round(sleep_deep_hours, 2)
        factors["deep_sleep_penalty"] = round(deep_sleep_penalty, 1)
    if resting_hr is not None:
        factors["resting_hr"] = resting_hr
        factors["resting_hr_penalty"] = round(resting_hr_penalty, 1)

    return {
        "readiness_score": score,
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "tier_color": TIER_COLORS[tier],
        "tier_description": TIER_DESCRIPTIONS[tier],
        "factors": factors,
        "sleep_data_available": sleep_hours is not None,
    }
