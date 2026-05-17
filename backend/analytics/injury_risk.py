"""
Injury risk assessment engine.

Based on:
- Gabbett (2016) ACWR model — Br J Sports Med
- Banister monotony/strain research
- 10% weekly mileage increase rule (Buist et al., 2008)

All deterministic — no LLM.
"""
from __future__ import annotations
from typing import Any, Dict, List

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

_RISK_LABELS = {
    RISK_LOW: "🟢 Low",
    RISK_MEDIUM: "🟡 Medium",
    RISK_HIGH: "🔴 High",
}

_RISK_COLORS = {
    RISK_LOW: "#48BB78",
    RISK_MEDIUM: "#ECC94B",
    RISK_HIGH: "#F56565",
}


def compute_injury_risk(
    acwr: float,
    monotony_score: float,
    consecutive_training_days: int,
    load_change_pct: float,
    recovery_debt: float,
    recovery_score: float,
    fatigue_score: float,
) -> Dict[str, Any]:
    """
    Assess injury risk level and explain contributing factors.

    Parameters
    ----------
    acwr                     : Acute:Chronic Workload Ratio (7d / 28d avg)
    monotony_score           : Banister monotony (mean/std daily load 28d)
    consecutive_training_days: current unbroken training streak
    load_change_pct          : % change in load this week vs last week
    recovery_debt            : (ATL-CTL)/CTL
    recovery_score           : 0–100 (higher = better recovered)
    fatigue_score            : 0–100 (higher = more fatigued)

    Returns
    -------
    Dict with risk, risk_label, risk_color, risk_points, factors, recommendations
    """
    factors: List[str] = []
    risk_points = 0
    recommendations: List[str] = []

    # ── ACWR (primary indicator) ──────────────────────────────────────────
    if acwr > 1.5:
        factors.append(
            f"Very high ACWR ({acwr:.2f}) — acute load is {acwr:.1f}× your chronic baseline. "
            "Research shows >1.5 significantly elevates soft-tissue injury risk."
        )
        risk_points += 40
        recommendations.append("Reduce this week's load by 25–30% immediately.")
    elif acwr > 1.3:
        factors.append(
            f"Elevated ACWR ({acwr:.2f}) — workload is ramping above the safe 1.0–1.3 zone."
        )
        risk_points += 20
        recommendations.append("Avoid adding any additional intensity this week.")

    # ── Weekly load spike ─────────────────────────────────────────────────
    if load_change_pct > 30:
        factors.append(
            f"Load spike: +{load_change_pct:.0f}% vs last week. "
            "The 10% weekly increase rule is widely supported to prevent overuse injuries."
        )
        risk_points += 25
        recommendations.append("Keep next week's mileage within 10% of the previous week.")
    elif load_change_pct > 15:
        factors.append(f"Moderate load increase: +{load_change_pct:.0f}% vs last week.")
        risk_points += 8

    # ── Monotony ──────────────────────────────────────────────────────────
    if monotony_score > 2.5:
        factors.append(
            f"High training monotony ({monotony_score:.1f}) — daily load has little variation. "
            "Constant stress without recovery prevents tissue adaptation."
        )
        risk_points += 15
        recommendations.append("Add a hard/easy day structure — vary effort between sessions.")
    elif monotony_score > 2.0:
        factors.append(f"Moderate monotony ({monotony_score:.1f}) — consider adding more load variation.")
        risk_points += 5

    # ── Consecutive days ──────────────────────────────────────────────────
    if consecutive_training_days >= 7:
        factors.append(
            f"{consecutive_training_days} consecutive training days — no full rest in over a week. "
            "Connective tissue requires 48+ hrs between hard stimuli to repair."
        )
        risk_points += 20
        recommendations.append("Take a rest or active-recovery day today or tomorrow.")
    elif consecutive_training_days >= 5:
        factors.append(
            f"{consecutive_training_days} consecutive days — a rest day is due soon."
        )
        risk_points += 8

    # ── Recovery ─────────────────────────────────────────────────────────
    if recovery_score < 30:
        factors.append(
            f"Very low recovery score ({recovery_score:.0f}/100) — body has insufficient "
            "time to repair micro-tears from recent training."
        )
        risk_points += 15
        recommendations.append("Prioritise sleep (8+ hrs), protein, and hydration over the next 48 hrs.")
    elif recovery_score < 45:
        factors.append(f"Low recovery score ({recovery_score:.0f}/100).")
        risk_points += 7

    # ── Recovery debt ─────────────────────────────────────────────────────
    if recovery_debt > 0.5:
        factors.append(
            f"High recovery debt ({recovery_debt:.2f}) — you've been training significantly "
            "above your chronic baseline for an extended period."
        )
        risk_points += 10

    # ── Determine overall risk ────────────────────────────────────────────
    if risk_points >= 40:
        risk = RISK_HIGH
    elif risk_points >= 15:
        risk = RISK_MEDIUM
    else:
        risk = RISK_LOW

    if not factors:
        summary = "No significant injury risk factors detected. Keep up the consistent training."
    else:
        summary = f"{len(factors)} risk factor(s) identified. See details below."

    if not recommendations:
        recommendations.append("Maintain current load structure. Continue with planned training.")

    return {
        "risk": risk,
        "risk_label": _RISK_LABELS[risk],
        "risk_color": _RISK_COLORS[risk],
        "risk_points": risk_points,
        "factors": factors,
        "recommendations": recommendations,
        "summary": summary,
    }
