"""
Enhanced race prediction engine.

Uses the Riegel endurance formula as the base, then applies fitness trajectory
modifiers derived from HR efficiency, pace efficiency, consistency trends, and
VO2 max (from Garmin) for calibrated marathon projections.

Produces:
- Projected finish times for 5K, 10K, HM, Marathon
- Confidence score per distance
- Overall fitness trajectory direction
"""
from __future__ import annotations
import math
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# VO2max → marathon time lookup (Jack Daniels VDOT table, seconds)
# ---------------------------------------------------------------------------
_VDOT_MARATHON: list[tuple[float, float]] = [
    (28, 5 * 3600 + 29 * 60 + 39),
    (30, 5 * 3600 + 10 * 60 + 53),
    (32, 4 * 3600 + 54 * 60 + 6),
    (34, 4 * 3600 + 38 * 60 + 50),
    (36, 4 * 3600 + 25 * 60 + 0),
    (38, 4 * 3600 + 12 * 60 + 17),
    (40, 4 * 3600 + 0 * 60 + 32),
    (42, 3 * 3600 + 49 * 60 + 41),
    (44, 3 * 3600 + 39 * 60 + 38),
    (46, 3 * 3600 + 30 * 60 + 18),
    (48, 3 * 3600 + 21 * 60 + 35),
    (50, 3 * 3600 + 13 * 60 + 23),
    (52, 3 * 3600 + 5 * 60 + 42),
    (55, 2 * 3600 + 54 * 60 + 29),
    (60, 2 * 3600 + 38 * 60 + 0),
    (65, 2 * 3600 + 23 * 60 + 22),
    (70, 2 * 3600 + 10 * 60 + 20),
]


def _vo2max_to_marathon_sec(vo2max: float) -> float:
    """Interpolate marathon time (seconds) from VO2max using Daniels VDOT table."""
    vdots = [v for v, _ in _VDOT_MARATHON]
    times = [t for _, t in _VDOT_MARATHON]
    if vo2max <= vdots[0]:
        return times[0]
    if vo2max >= vdots[-1]:
        return times[-1]
    for i in range(len(vdots) - 1):
        if vdots[i] <= vo2max <= vdots[i + 1]:
            frac = (vo2max - vdots[i]) / (vdots[i + 1] - vdots[i])
            return times[i] + frac * (times[i + 1] - times[i])
    return float("nan")


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
    vo2max: float | None = None,
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
    vo2max                 : VO2 max from Garmin (mL/kg/min) — used to calibrate marathon estimate

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

    # VO2 max fitness adjustment (Garmin data)
    if vo2max is not None:
        if vo2max >= 52:
            modifier -= 0.020
        elif vo2max >= 47:
            modifier -= 0.010
        elif vo2max < 38:
            modifier += 0.020
        elif vo2max < 42:
            modifier += 0.010

    modifier = max(-0.05, min(0.05, modifier))  # cap at ±5%

    # ── Confidence score ──────────────────────────────────────────────────
    # Weighted: consistency (40%) + aerobic fitness data (30%) + freshness (30%)
    freshness = max(0.0, 100.0 - fatigue_score)
    confidence = (
        consistency_score * 0.40
        + aerobic_fitness_score * 0.30
        + freshness * 0.30
    )
    # VO2 max boosts confidence (adds up to 8 points when available)
    if vo2max is not None:
        confidence = min(100.0, confidence + 8.0)
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

        # 3. For marathon: fall back to VO2max VDOT estimate when no PR/projection is available
        if base_sec is None and label == "marathon" and vo2max is not None:
            vo2_est = _vo2max_to_marathon_sec(vo2max)
            if not math.isnan(vo2_est):
                base_sec = vo2_est
                source = "vo2max estimate"

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
            if source == "vo2max estimate":
                dist_confidence *= 0.65  # VO2max estimate less certain than Riegel
            else:
                dist_confidence *= 0.75  # marathon is harder to predict

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
        "vo2max_used": vo2max,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
