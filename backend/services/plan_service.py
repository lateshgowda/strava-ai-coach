"""FM Plan Tracker service — parse xlsx, match activities, compute status."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.activity import Activity
from backend.models.training_plan import AIMonthSummary, AIRunReview, AIWeeklySummary, TrainingPlan

PLAN_FILE = Path("/data/Latesh_FM_Plan.xlsx")

_RUN_TYPES = {"Run", "TrailRun", "VirtualRun", "Race"}


def _parse_distance_km(raw: str) -> Optional[float]:
    """Parse "8K", "21.1K", "42.2K", or plain float string → km float."""
    if not raw or str(raw).strip().lower() in ("nan", ""):
        return None
    s = str(raw).strip().upper()
    m = re.match(r"([\d.]+)\s*K?$", s)
    if m:
        return float(m.group(1))
    return None


def _to_iso_week(d: date) -> str:
    return d.strftime("%G-W%V")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_plan(db: Session, xlsx_path: Path = PLAN_FILE) -> Dict[str, int]:
    """Parse the FM plan xlsx and upsert all workouts. Returns import stats."""
    df = pd.read_excel(str(xlsx_path), header=0)

    inserted = updated = skipped = 0

    for _, row in df.iterrows():
        raw_date = row.iloc[0]
        if pd.isna(raw_date):
            skipped += 1
            continue

        try:
            if isinstance(raw_date, datetime):
                plan_date = raw_date.date()
            elif isinstance(raw_date, date):
                plan_date = raw_date
            else:
                plan_date = pd.to_datetime(str(raw_date)).date()
        except Exception:
            skipped += 1
            continue

        raw_dist = row.iloc[1] if len(row) > 1 else None
        dist_km = _parse_distance_km(str(raw_dist) if not pd.isna(raw_dist) else "")
        if dist_km is None:
            skipped += 1
            continue

        raw_details = row.iloc[2] if len(row) > 2 else None
        details = str(raw_details).strip() if raw_details is not None and not pd.isna(raw_details) else None
        if details in ("nan", ""):
            details = None

        raw_session = row.iloc[3] if len(row) > 3 else None
        session_type = (
            str(raw_session).strip()
            if raw_session is not None and not pd.isna(raw_session)
            else "Run"
        )
        if session_type in ("nan", ""):
            session_type = "Run"

        weekly_volume = None
        if len(row) > 4 and not pd.isna(row.iloc[4]):
            raw_vol = row.iloc[4]
            try:
                v_str = str(raw_vol).strip().upper().replace("K", "")
                weekly_volume = float(v_str)
            except Exception:
                pass

        iso_week = _to_iso_week(plan_date)

        existing = db.query(TrainingPlan).filter(TrainingPlan.plan_date == plan_date).first()
        if existing:
            existing.distance_km = dist_km
            existing.session_type = session_type
            existing.details = details
            existing.weekly_volume_km = weekly_volume
            existing.iso_week = iso_week
            updated += 1
        else:
            db.add(TrainingPlan(
                plan_date=plan_date,
                distance_km=dist_km,
                session_type=session_type,
                details=details,
                weekly_volume_km=weekly_volume,
                iso_week=iso_week,
            ))
            inserted += 1

    db.commit()
    return {"inserted": inserted, "updated": updated, "skipped": skipped, "total": inserted + updated}


# ---------------------------------------------------------------------------
# Activity matching
# ---------------------------------------------------------------------------


def find_matching_activity(db: Session, plan_date: date, dist_km: float) -> Optional[Activity]:
    """Find a Run activity within ±1 day of plan_date, preferring closest distance."""
    window_start = datetime.combine(plan_date - timedelta(days=1), datetime.min.time())
    window_end = datetime.combine(plan_date + timedelta(days=1), datetime.max.time())

    candidates = (
        db.query(Activity)
        .filter(
            Activity.activity_type.in_(list(_RUN_TYPES)),
            Activity.start_date >= window_start,
            Activity.start_date <= window_end,
        )
        .all()
    )

    if not candidates:
        return None

    def dist_diff(a: Activity) -> float:
        return abs((a.distance or 0) / 1000.0 - dist_km)

    return min(candidates, key=dist_diff)


def workout_status(workout: TrainingPlan, matched: Optional[Activity]) -> str:
    if matched:
        return "completed"
    if date.today() > workout.plan_date + timedelta(days=1):
        return "missed"
    return "upcoming"


def _format_pace(moving_time: Optional[int], distance: Optional[float]) -> Optional[str]:
    if not moving_time or not distance or distance <= 0:
        return None
    pace_raw = (moving_time / 60.0) / (distance / 1000.0)
    mins = int(pace_raw)
    secs = int((pace_raw - mins) * 60)
    return f"{mins}:{secs:02d}"


# ---------------------------------------------------------------------------
# Week detail
# ---------------------------------------------------------------------------


def get_week_data(db: Session, iso_week: str) -> Dict[str, Any]:
    """Return full week data: planned workouts with matched activities and status."""
    workouts = (
        db.query(TrainingPlan)
        .filter(TrainingPlan.iso_week == iso_week)
        .order_by(TrainingPlan.plan_date)
        .all()
    )

    result_workouts = []
    total_planned_km = 0.0
    total_actual_km = 0.0

    for w in workouts:
        matched = find_matching_activity(db, w.plan_date, w.distance_km)
        status = workout_status(w, matched)
        total_planned_km += w.distance_km

        actual_data = None
        if matched:
            dist_km = round((matched.distance or 0) / 1000.0, 2)
            total_actual_km += dist_km
            actual_data = {
                "strava_id": matched.strava_id,
                "date": matched.start_date.strftime("%Y-%m-%d"),
                "name": matched.name,
                "distance_km": dist_km,
                "avg_hr": round(matched.average_heartrate) if matched.average_heartrate else None,
                "pace": _format_pace(matched.moving_time, matched.distance),
                "spm": round(matched.average_cadence) if matched.average_cadence else None,
            }

        saved_review = None
        if matched:
            rev = db.query(AIRunReview).filter(AIRunReview.strava_id == matched.strava_id).first()
            if rev:
                saved_review = rev.review_text

        result_workouts.append({
            "id": w.id,
            "plan_date": w.plan_date.isoformat(),
            "distance_km": w.distance_km,
            "session_type": w.session_type,
            "details": w.details,
            "status": status,
            "actual": actual_data,
            "ai_review": saved_review,
        })

    summary_obj = db.query(AIWeeklySummary).filter(AIWeeklySummary.iso_week == iso_week).first()

    return {
        "iso_week": iso_week,
        "workouts": result_workouts,
        "planned_km": round(total_planned_km, 1),
        "actual_km": round(total_actual_km, 1),
        "weekly_summary": summary_obj.summary_text if summary_obj else None,
    }


# ---------------------------------------------------------------------------
# Weeks list (lean — no activity matching, just plan data)
# ---------------------------------------------------------------------------


def get_all_weeks(db: Session) -> List[Dict[str, Any]]:
    """List all ISO weeks with planned info. Status counts computed per request."""
    rows = (
        db.query(
            TrainingPlan.iso_week,
            func.count(TrainingPlan.id).label("total"),
            func.sum(TrainingPlan.distance_km).label("planned_km"),
            func.min(TrainingPlan.plan_date).label("week_start"),
        )
        .group_by(TrainingPlan.iso_week)
        .order_by(func.min(TrainingPlan.plan_date))
        .all()
    )

    today = date.today()
    weeks = []
    for row in rows:
        week_start = row.week_start
        week_end = week_start + timedelta(days=6)

        if week_end < today - timedelta(days=1):
            phase = "past"
        elif week_start <= today <= week_end + timedelta(days=1):
            phase = "current"
        else:
            phase = "future"

        weeks.append({
            "iso_week": row.iso_week,
            "week_start": row.week_start.isoformat(),
            "total_workouts": int(row.total),
            "planned_km": round(float(row.planned_km), 1),
            "phase": phase,
        })

    return weeks


# ---------------------------------------------------------------------------
# Plan status (summary stats — fast)
# ---------------------------------------------------------------------------


def get_plan_status(db: Session) -> Dict[str, Any]:
    """High-level plan stats without full activity matching."""
    total = db.query(TrainingPlan).count()
    if total == 0:
        return {"total_workouts": 0, "total_weeks": 0, "total_planned_km": 0.0}

    row = db.query(
        func.sum(TrainingPlan.distance_km).label("total_km"),
        func.count(func.distinct(TrainingPlan.iso_week)).label("total_weeks"),
    ).one()

    return {
        "total_workouts": total,
        "total_weeks": int(row.total_weeks),
        "total_planned_km": round(float(row.total_km), 1),
    }


# ---------------------------------------------------------------------------
# AI review persistence
# ---------------------------------------------------------------------------


def save_run_review(db: Session, strava_id: int, plan_workout_id: Optional[int], text: str) -> None:
    existing = db.query(AIRunReview).filter(AIRunReview.strava_id == strava_id).first()
    if existing:
        existing.review_text = text
        existing.generated_at = datetime.utcnow()
    else:
        db.add(AIRunReview(
            strava_id=strava_id,
            plan_workout_id=plan_workout_id,
            review_text=text,
        ))
    db.commit()


def save_weekly_summary(db: Session, iso_week: str, text: str) -> None:
    existing = db.query(AIWeeklySummary).filter(AIWeeklySummary.iso_week == iso_week).first()
    if existing:
        existing.summary_text = text
        existing.generated_at = datetime.utcnow()
    else:
        db.add(AIWeeklySummary(iso_week=iso_week, summary_text=text))
    db.commit()


# ---------------------------------------------------------------------------
# Monthly data
# ---------------------------------------------------------------------------


def get_all_months(db: Session) -> List[Dict[str, Any]]:
    """List all distinct months that have plan workouts."""
    import calendar

    dates = db.query(TrainingPlan.plan_date).distinct().order_by(TrainingPlan.plan_date).all()

    seen: set = set()
    today = date.today()
    result = []
    for row in dates:
        d = row.plan_date
        key = f"{d.year:04d}-{d.month:02d}"
        if key in seen:
            continue
        seen.add(key)
        last_day = calendar.monthrange(d.year, d.month)[1]
        month_end = date(d.year, d.month, last_day)
        if month_end < today:
            phase = "past"
        elif d.year == today.year and d.month == today.month:
            phase = "current"
        else:
            phase = "future"
        result.append({
            "year_month": key,
            "display": date(d.year, d.month, 1).strftime("%B %Y"),
            "phase": phase,
        })
    return result


def get_month_data(db: Session, year_month: str) -> Dict[str, Any]:
    """Return full month data: all workouts, per-week chart data, and saved summary."""
    import calendar

    year, month = int(year_month[:4]), int(year_month[5:7])
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])

    workouts = (
        db.query(TrainingPlan)
        .filter(TrainingPlan.plan_date >= start, TrainingPlan.plan_date <= end)
        .order_by(TrainingPlan.plan_date)
        .all()
    )

    result_workouts = []
    total_planned_km = 0.0
    total_actual_km = 0.0
    completed = missed = upcoming = 0
    week_buckets: Dict[str, Dict[str, float]] = {}

    for w in workouts:
        matched = find_matching_activity(db, w.plan_date, w.distance_km)
        status = workout_status(w, matched)
        total_planned_km += w.distance_km

        actual_data = None
        if matched:
            dist_km = round((matched.distance or 0) / 1000.0, 2)
            total_actual_km += dist_km
            actual_data = {
                "strava_id": matched.strava_id,
                "date": matched.start_date.strftime("%Y-%m-%d"),
                "distance_km": dist_km,
                "avg_hr": round(matched.average_heartrate) if matched.average_heartrate else None,
                "pace": _format_pace(matched.moving_time, matched.distance),
                "spm": round(matched.average_cadence) if matched.average_cadence else None,
            }
            completed += 1
        elif status == "missed":
            missed += 1
        else:
            upcoming += 1

        bkt = week_buckets.setdefault(w.iso_week, {"planned_km": 0.0, "actual_km": 0.0})
        bkt["planned_km"] += w.distance_km
        if actual_data:
            bkt["actual_km"] += actual_data["distance_km"]

        saved_review = None
        if matched:
            rev = db.query(AIRunReview).filter(AIRunReview.strava_id == matched.strava_id).first()
            if rev:
                saved_review = rev.review_text

        result_workouts.append({
            "id": w.id,
            "plan_date": w.plan_date.isoformat(),
            "distance_km": w.distance_km,
            "session_type": w.session_type,
            "details": w.details,
            "iso_week": w.iso_week,
            "status": status,
            "actual": actual_data,
            "ai_review": saved_review,
        })

    week_chart = [
        {"week": k, "planned_km": round(v["planned_km"], 1), "actual_km": round(v["actual_km"], 1)}
        for k, v in sorted(week_buckets.items())
    ]

    total = completed + missed + upcoming
    summary_obj = db.query(AIMonthSummary).filter(AIMonthSummary.month_key == year_month).first()

    # Month number in training plan (Sep 2026 = month 1)
    plan_start = date(2026, 9, 1)
    month_num = (year - plan_start.year) * 12 + (month - plan_start.month) + 1

    return {
        "year_month": year_month,
        "display": date(year, month, 1).strftime("%B %Y"),
        "month_num": month_num,
        "workouts": result_workouts,
        "planned_km": round(total_planned_km, 1),
        "actual_km": round(total_actual_km, 1),
        "completed": completed,
        "missed": missed,
        "upcoming": upcoming,
        "completion_pct": round((completed / total) * 100) if total > 0 else 0,
        "week_chart": week_chart,
        "monthly_summary": summary_obj.summary_text if summary_obj else None,
    }


def save_month_summary(db: Session, year_month: str, text: str) -> None:
    existing = db.query(AIMonthSummary).filter(AIMonthSummary.month_key == year_month).first()
    if existing:
        existing.summary_text = text
        existing.generated_at = datetime.utcnow()
    else:
        db.add(AIMonthSummary(month_key=year_month, summary_text=text))
    db.commit()
