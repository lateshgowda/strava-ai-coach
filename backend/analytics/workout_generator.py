"""
Structured workout generator.

Generates concrete training sessions based on athlete fitness state and goal.
All paces are derived from recent average easy pace using scientifically-grounded
pace relationships.

Pace relationships (Jack Daniels VDOT model approximations):
  Recovery pace    : easy_pace × 1.18
  Easy pace        : ~recent avg pace + 5-8% slower
  Marathon pace    : easy_pace × 0.93
  Threshold pace   : easy_pace × 0.87 (lactate threshold, ~60 min max)
  10K pace         : easy_pace × 0.82
  5K pace          : easy_pace × 0.78
  Interval pace    : easy_pace × 0.75 (VO2max effort)
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


def _pace_str(min_per_km: float) -> str:
    """Format a pace in min/km as 'M:SS/km'."""
    if not min_per_km or math.isnan(min_per_km) or min_per_km <= 0:
        return "–"
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    return f"{m}:{s:02d}/km"


def _hr_zone(max_hr: Optional[int], pct_low: float, pct_high: float) -> str:
    if not max_hr or max_hr <= 0:
        return f"Zone target: {int(pct_low*100)}–{int(pct_high*100)}% max HR"
    return f"{int(max_hr*pct_low)}–{int(max_hr*pct_high)} bpm"


def generate_workout(
    workout_type: str,                   # "easy" | "intervals" | "threshold" | "long_run" | "recovery"
    recent_avg_pace: Optional[float],    # min/km from last 4 eligible runs
    fatigue_score: float,
    readiness_score: float,
    goal: Optional[str] = None,          # "hm" | "10k" | "marathon" | "base_building"
    available_minutes: int = 60,
    max_hr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns a fully structured workout with warmup, blocks, cooldown, and coaching notes.
    """
    base_pace = recent_avg_pace if (recent_avg_pace and not math.isnan(recent_avg_pace) and recent_avg_pace > 0) else 6.5

    # Derived training paces
    recovery_pace = base_pace * 1.18
    easy_pace = base_pace * 1.06
    marathon_pace = base_pace * 0.93
    threshold_pace = base_pace * 0.87
    ten_k_pace = base_pace * 0.82
    five_k_pace = base_pace * 0.78
    interval_pace = base_pace * 0.75

    goal_norm = (goal or "base_building").lower()
    workout: Dict[str, Any] = {}

    # ── Recovery Run ─────────────────────────────────────────────────────
    if workout_type == "recovery":
        run_min = min(30, available_minutes - 5)
        workout = {
            "name": "Recovery Run",
            "type": "recovery",
            "target_hr_zone": _hr_zone(max_hr, 0.60, 0.70),
            "warmup": "5 min walk — let your body wake up",
            "blocks": [
                {
                    "description": "Very easy jog — aerobic flush",
                    "duration_min": run_min,
                    "target_pace": _pace_str(recovery_pace),
                    "effort": "Very easy. If it feels too slow, it is correct.",
                },
            ],
            "cooldown": "5 min walk + light dynamic stretching",
            "estimated_distance_km": round(run_min / recovery_pace, 1),
            "coaching_note": (
                "Recovery runs enhance blood flow and clear lactate without adding training stress. "
                "The goal is circulation, not fitness. Run by feel — slower than easy."
            ),
        }

    # ── Easy Run ─────────────────────────────────────────────────────────
    elif workout_type == "easy":
        run_min = min(50, available_minutes - 15)
        workout = {
            "name": "Easy Aerobic Run",
            "type": "easy",
            "target_hr_zone": _hr_zone(max_hr, 0.70, 0.80),
            "warmup": "First 10 min: ramp into pace from jog",
            "blocks": [
                {
                    "description": "Steady easy run — Zone 2",
                    "duration_min": run_min,
                    "target_pace": _pace_str(easy_pace),
                    "effort": "Conversational — you can speak full sentences comfortably",
                },
            ],
            "cooldown": "5 min walk + light stretching",
            "estimated_distance_km": round(run_min / easy_pace, 1),
            "coaching_note": (
                "80% of your weekly running should feel this easy. Zone 2 running builds "
                "mitochondrial density and fat oxidation — the aerobic base everything else depends on. "
                "Most runners run easy runs too fast."
            ),
        }

    # ── Interval Session ────────────────────────────────────────────────
    elif workout_type == "intervals":
        warmup_min = 15
        cooldown_min = 15
        work_min = available_minutes - warmup_min - cooldown_min

        if goal_norm in ("hm", "half_marathon"):
            reps, rep_label, rep_pace, rest_sec = 6, "1 km", five_k_pace, 90
            session_focus = "VO2max development for HM speed"
        elif goal_norm == "10k":
            reps, rep_label, rep_pace, rest_sec = 8, "600 m", interval_pace, 90
            session_focus = "VO2max and neuromuscular speed for 10K"
        elif goal_norm == "marathon":
            reps, rep_label, rep_pace, rest_sec = 5, "1.5 km", ten_k_pace, 120
            session_focus = "Lactate threshold development for marathon"
        else:
            reps, rep_label, rep_pace, rest_sec = 5, "1 km", five_k_pace, 120
            session_focus = "General aerobic capacity"

        workout = {
            "name": "Interval Session",
            "type": "intervals",
            "target_hr_zone": _hr_zone(max_hr, 0.85, 0.95),
            "warmup": f"{warmup_min} min easy jog + 4 × 20 sec strides with full recovery",
            "blocks": [
                {
                    "description": f"{reps} × {rep_label} at 5K effort",
                    "reps": reps,
                    "rep_distance": rep_label,
                    "target_pace": _pace_str(rep_pace),
                    "rest": f"{rest_sec} sec easy jog recovery between reps",
                    "effort": "Hard but controlled — 1-word answers only. Each rep should feel the same.",
                },
            ],
            "cooldown": f"{cooldown_min} min easy jog + full stretching routine",
            "session_focus": session_focus,
            "coaching_note": (
                f"First rep should feel almost too easy — resist going out hard. "
                f"If pace drops > 5% from rep 1 by rep {reps}, stop and cool down. "
                "Quality over quantity: 5 sharp reps beat 8 ragged ones."
            ),
        }

    # ── Threshold Run ────────────────────────────────────────────────────
    elif workout_type == "threshold":
        warmup_min = 15
        cooldown_min = 10
        tempo_min = min(35, available_minutes - warmup_min - cooldown_min)
        tempo_min = max(15, tempo_min)

        workout = {
            "name": "Threshold / Tempo Run",
            "type": "threshold",
            "target_hr_zone": _hr_zone(max_hr, 0.80, 0.90),
            "warmup": f"{warmup_min} min easy jog",
            "blocks": [
                {
                    "description": f"{tempo_min} min continuous tempo at lactate threshold",
                    "duration_min": tempo_min,
                    "target_pace": _pace_str(threshold_pace),
                    "effort": (
                        "Comfortably hard — 1–2 word answers, sustained for 60 min but not comfortable. "
                        "This is the pace just below where lactate starts to accumulate rapidly."
                    ),
                },
            ],
            "cooldown": f"{cooldown_min} min easy jog + stretching",
            "estimated_distance_km": round(tempo_min / threshold_pace, 1),
            "coaching_note": (
                "Threshold training raises your lactate turn-point — the single biggest driver "
                "of endurance race performance. 2× per week maximum. If HR exceeds "
                "90% max HR, you've gone too hard and need to back off."
            ),
        }

    # ── Long Run ─────────────────────────────────────────────────────────
    elif workout_type == "long_run":
        base_min = min(120, available_minutes - 20)
        finish_min = min(15, base_min // 6)
        main_min = base_min - finish_min

        blocks: List[Dict[str, Any]] = [
            {
                "description": "Long easy run — conversational pace",
                "duration_min": main_min,
                "target_pace": _pace_str(easy_pace * 1.05),
                "effort": "Easy aerobic — you should never be breathing hard. Err on the slow side.",
            },
        ]
        if goal_norm in ("hm", "half_marathon", "marathon") and finish_min >= 10:
            blocks.append({
                "description": "Optional: finish at marathon/HM pace",
                "duration_min": finish_min,
                "target_pace": _pace_str(marathon_pace),
                "effort": "Moderate controlled effort — practise running on tired legs",
            })

        workout = {
            "name": "Long Run",
            "type": "long_run",
            "target_hr_zone": _hr_zone(max_hr, 0.65, 0.78),
            "warmup": "First 2 km walk/very easy jog to protect joints",
            "blocks": blocks,
            "cooldown": "5 min walk + foam rolling (especially quads and calves)",
            "estimated_distance_km": round(base_min / (easy_pace * 1.05), 1),
            "coaching_note": (
                "Long runs build mitochondrial density, fat oxidation, and psychological endurance. "
                "Fuel: 30–60 g carbs/hr if run exceeds 75 min. Hydrate every 20 min. "
                "Aim for even or slightly negative splits — positive splits mean too fast a start."
            ),
        }

    else:
        workout = {
            "name": "Easy Run",
            "type": "easy",
            "blocks": [
                {
                    "description": "Easy run",
                    "duration_min": available_minutes,
                    "target_pace": _pace_str(easy_pace),
                    "effort": "Conversational",
                },
            ],
            "coaching_note": "Default easy day.",
        }

    # ── Readiness warning ─────────────────────────────────────────────────
    if readiness_score < 45 and workout_type in ("intervals", "threshold"):
        workout["readiness_warning"] = (
            f"⚠️ Readiness score is {readiness_score:.0f}/100. "
            "Consider substituting with an easy run or recovery session today. "
            "Performing quality work on low readiness risks poor adaptation and injury."
        )

    workout["based_on_pace"] = _pace_str(base_pace)
    return workout
