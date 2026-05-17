"""
Enhanced race prediction engine.

Uses the Riegel endurance formula as the base, then applies fitness trajectory
modifiers derived from HR efficiency, pace efficiency, and consistency trends.

Produces:
- Projected finish times for 5K, 10K, HM, Marathon
- Confidence score per distance
- Overall fitness trajectory direction
"""
from __future__ import annotations
import math
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Core Riegel formula
# ---------------------------------------------------------------------------

def riegel_time(actual_dist_km: float, actual_time_sec: float, target_dist_km: float) -> float:
    """
    Riegel endurance formula: T2 = T1 × (D2/D1)^1.06
    Returns predicted time in seconds, or nan on invalid input.
    """
    if actual_dist_km <= 0 or actual_time_sec <= 0 or target_dist_km <= 0:
        return float("nan")
    return actual_time_sec * (target_dist_km / actual_dist_km) ** 1.06


def format_duration(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

def compute_race_predictions(
    personal_records: Dict[str, Any],
    consistency_score: float,
    aerobic_fitness_score: float,
    fatigue_score: float,
    hr_efficiency_trend: str,
    pace_efficiency_trend: str,
    acwr: float,
) -> Dict[str, Any]:
    """
    Compute race-time projections with confidence scores and fitness trajectory.

    The Riegel time is the baseline.  A trajectory modifier adjusts it based
    on fitness trends — improving trends yield slightly faster projections,
    declining trends yield slower.

    Parameters
    ----------
    personal_records       : dict with keys '5k', '10k', 'half_marathon' (seconds)
    consistency_score      : 0–100 (% of last 8 weeks with runs)
    aerobic_fitness_score  : 0–100 (pace-HR efficiency proxy)
    fatigue_score          : 0–100 (higher = more fatigued)
    hr_efficiency_trend    : 'improving' | 'stable' | 'declining'
    pace_efficiency_trend  : 'improving' | 'stable' | 'declining'
    acwr                   : current acute:chronic workload ratio

    Returns
    -------
    Dict with predictions (per distance), overall_confidence, fitness_trajectory
    """
    # ── Trajectory modifier ───────────────────────────────────────────────
    # Improving efficiency trends → slightly faster projections (−3% max)
    # Declining or fatigued → slightly slower (+5% max)
    modifier = 0.0

    # Efficiency trends
    if hr_efficiency_trend == "improving" and pace_efficiency_trend == "improving":
        modifier -= 0.030
    elif hr_efficiency_trend == "improving" or pace_efficiency_trend == "improving":
        modifier -= 0.015
    elif hr_efficiency_trend == "declining" and pace_efficiency_trend == "declining":
        modifier += 0.025
    elif hr_efficiency_trend == "declining" or pace_efficiency_trend == "declining":
        modifier += 0.012

    # Fatigue adjustment (predicting a race when fatigued → slower)
    if fatigue_score > 75:
        modifier += 0.030
    elif fatigue_score > 60:
        modifier += 0.015

    # ACWR adjustment
    if acwr > 1.3:
        modifier += 0.015

    modifier = max(-0.05, min(0.05, modifier))  # cap at ±5%

    # ── Confidence score ──────────────────────────────────────────────────
    # Weighted: consistency (40%) + aerobic fitness data (30%) + freshness (30%)
    freshness = max(0.0, 100.0 - fatigue_score)
    confidence = (
        consistency_score * 0.40
        + aerobic_fitness_score * 0.30
        + freshness * 0.30
    )
    confidence = round(min(100.0, max(0.0, confidence)), 0)

    # ── Fitness trajectory ─────────────────────────────────────────────────
    if modifier < -0.01:
        trajectory = "improving"
    elif modifier > 0.01:
        trajectory = "declining"
    else:
        trajectory = "stable"

    # ── Per-distance predictions ───────────────────────────────────────────
    distances = {
        "5k": 5.0,
        "10k": 10.0,
        "half_marathon": 21.0975,
        "marathon": 42.195,
    }

    base_prs: Dict[str, Optional[float]] = {
        "5k": _to_float(personal_records.get("5k")),
        "10k": _to_float(personal_records.get("10k")),
        "half_marathon": _to_float(personal_records.get("half_marathon")),
    }

    predictions: Dict[str, Any] = {}

    for label, target_km in distances.items():
        # 1. Try direct PR
        base_sec = base_prs.get(label)
        source = label if base_sec else None

        # 2. Project from best shorter-distance PR
        if base_sec is None:
            for src_label, src_km in [("5k", 5.0), ("10k", 10.0), ("half_marathon", 21.0975)]:
                if src_km < target_km and base_prs.get(src_label):
                    est = riegel_time(src_km, base_prs[src_label], target_km)  # type: ignore[arg-type]
                    if not math.isnan(est):
                        base_sec = est
                        source = f"projected from {src_label}"
                        break

        if base_sec is None or math.isnan(base_sec):
            predictions[label] = {
                "time_str": "–",
                "time_sec": None,
                "confidence": 0,
                "trajectory": "unknown",
                "source": None,
            }
            continue

        adjusted_sec = base_sec * (1.0 + modifier)

        # Distance-specific confidence adjustments
        dist_confidence = float(confidence)
        if source and "projected" in str(source):
            dist_confidence *= 0.80   # less confident for projections
        if label == "marathon":
            dist_confidence *= 0.75   # marathon is harder to predict

        predictions[label] = {
            "time_str": format_duration(int(adjusted_sec)),
            "time_sec": round(adjusted_sec),
            "confidence": round(min(100, max(0, dist_confidence))),
            "trajectory": trajectory,
            "source": source,
        }

    return {
        "predictions": predictions,
        "overall_confidence": int(confidence),
        "fitness_trajectory": trajectory,
        "trajectory_modifier_pct": round(modifier * 100, 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
