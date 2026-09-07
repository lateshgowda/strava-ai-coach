from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.models.activity import Activity, StravaToken


# ---------------------------------------------------------------------------
# Token CRUD
# ---------------------------------------------------------------------------


def save_token(db: Session, token_data: Dict) -> StravaToken:
    """Upsert the Strava OAuth token — delete any existing record first."""
    db.query(StravaToken).delete()
    db.flush()

    token = StravaToken(
        access_token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        expires_at=int(token_data["expires_at"]),
        athlete_id=int(token_data.get("athlete_id", 0)),
        athlete_name=token_data.get("athlete_name"),
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def get_token(db: Session) -> Optional[StravaToken]:
    """Return the stored OAuth token, or None if no token exists."""
    return db.query(StravaToken).first()


# ---------------------------------------------------------------------------
# Activity CRUD
# ---------------------------------------------------------------------------


def upsert_activity(db: Session, activity_data: Dict) -> Tuple[Activity, bool]:
    """
    Insert or update an Activity record.

    Returns a tuple of (Activity, is_new) where is_new is True when a new
    record was created and False when an existing record was updated.
    """
    strava_id: int = int(activity_data["strava_id"])
    existing = db.query(Activity).filter(Activity.strava_id == strava_id).first()

    if existing:
        for key, value in activity_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing, False

    activity = Activity(**{k: v for k, v in activity_data.items() if hasattr(Activity, k)})
    db.add(activity)
    db.commit()
    db.refresh(activity)
    return activity, True


def get_activities(
    db: Session,
    limit: int = 500,
    activity_type: Optional[str] = None,
) -> List[Activity]:
    """Return activities ordered by start_date descending."""
    query = db.query(Activity)
    if activity_type:
        query = query.filter(Activity.activity_type == activity_type)
    return (
        query.order_by(Activity.start_date.desc())
        .limit(limit)
        .all()
    )


def get_activity_by_strava_id(db: Session, strava_id: int) -> Optional[Activity]:
    """Fetch a single Activity by its Strava ID."""
    return (
        db.query(Activity)
        .filter(Activity.strava_id == strava_id)
        .first()
    )


def update_splits(db: Session, strava_id: int, splits_json: str) -> bool:
    """Cache splits_metric for an activity. Returns True if updated."""
    activity = get_activity_by_strava_id(db, strava_id)
    if not activity:
        return False
    activity.splits_metric = splits_json
    db.commit()
    return True


def get_latest_activity(db: Session) -> Optional[Activity]:
    """Return the most recently started activity."""
    return (
        db.query(Activity)
        .order_by(Activity.start_date.desc())
        .first()
    )


def get_activities_since(db: Session, since_date: datetime) -> List[Activity]:
    """Return all activities that started on or after *since_date*."""
    return (
        db.query(Activity)
        .filter(Activity.start_date >= since_date)
        .order_by(Activity.start_date.desc())
        .all()
    )


def delete_all_activities(db: Session) -> None:
    """Hard-delete every activity row — used before a full re-sync."""
    db.query(Activity).delete()
    db.commit()


# ---------------------------------------------------------------------------
# Athlete Profile CRUD
# ---------------------------------------------------------------------------


def get_profile(db: Session) -> Any:
    """Return the single athlete profile row, or None."""
    from backend.models.profile import AthleteProfile
    return db.query(AthleteProfile).first()


def save_profile(db: Session, data: Dict) -> Any:
    """Upsert the athlete profile (single-row table)."""
    from backend.models.profile import AthleteProfile
    existing = db.query(AthleteProfile).first()
    if existing:
        for k, v in data.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    p = AthleteProfile(**{k: v for k, v in data.items() if hasattr(AthleteProfile, k)})
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# Chat Memory CRUD
# ---------------------------------------------------------------------------


def save_chat_memory(db: Session, role: str, content: str) -> None:
    """Append a message to persistent chat history."""
    from backend.models.profile import ChatMemory
    db.add(ChatMemory(role=role, content=content))
    db.commit()


def get_recent_chat_memory(db: Session, limit: int = 20) -> List[Any]:
    """Return the most recent `limit` chat messages in chronological order."""
    from backend.models.profile import ChatMemory
    rows = (
        db.query(ChatMemory)
        .order_by(ChatMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def clear_chat_memory(db: Session) -> None:
    """Delete all stored chat history."""
    from backend.models.profile import ChatMemory
    db.query(ChatMemory).delete()
    db.commit()


# ---------------------------------------------------------------------------
# Daily Health Metrics CRUD
# ---------------------------------------------------------------------------


def _parse_date_flexible(value) -> "date":
    """Parse a date from various formats iOS Shortcuts may send."""
    from datetime import date as _date, datetime as _datetime
    import re

    if isinstance(value, _date) and not isinstance(value, _datetime):
        return value
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, str):
        # Normalise: collapse whitespace variants (narrow no-break space, etc.)
        s = re.sub(r"[\u00a0\u202f\u2009\u200b]+", " ", value.strip())
        # Try ISO first: "2026-05-17" or "2026-05-17T..."
        try:
            return _date.fromisoformat(s[:10])
        except ValueError:
            pass
        # iOS human format: "17 May 2026 at 3:09 PM" or "17 May 2026"
        for fmt in ("%d %B %Y at %I:%M %p", "%d %B %Y at %H:%M",
                    "%d %B %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return _datetime.strptime(s, fmt).date()
            except ValueError:
                pass
    # Fallback: use today — the Shortcut always runs on the current day
    from datetime import date as _date2
    return _date2.today()


def save_health_data(db: Session, data: Dict) -> Any:
    """Upsert a daily health metric row (one row per date)."""
    from backend.models.health import DailyHealthMetric

    row_date = _parse_date_flexible(data.get("date"))

    # Normalise sleep values:
    # - Shortcuts Calculate Statistics returns seconds; convert to hours if > 24
    # - Treat 0 as null (no samples found)
    # - Cap at 15h max (sanity check)
    _sleep_fields = ("sleep_duration_hours", "sleep_deep_hours",
                     "sleep_rem_hours", "sleep_core_hours", "sleep_awake_hours")
    for f in _sleep_fields:
        v = data.get(f)
        if v is None:
            continue
        if v == 0 or v == 0.0:
            data[f] = None
            continue
        # Convert seconds → hours if value looks like seconds (> 24)
        if v > 24:
            v = round(v / 3600, 2)
            data[f] = v
        # Sanity cap: more than 15h is still wrong data
        if v > 15:
            data[f] = None

    # If total sleep is missing but stages are present, derive it
    if not data.get("sleep_duration_hours"):
        stage_total = sum(
            data.get(f) or 0
            for f in ("sleep_deep_hours", "sleep_rem_hours", "sleep_core_hours")
        )
        if stage_total > 0:
            data["sleep_duration_hours"] = round(stage_total, 2)

    existing = db.query(DailyHealthMetric).filter(DailyHealthMetric.date == row_date).first()
    if existing:
        for k, v in data.items():
            if k != "date" and hasattr(existing, k):
                setattr(existing, k, v)
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing

    row = DailyHealthMetric(date=row_date, **{
        k: v for k, v in data.items()
        if k != "date" and hasattr(DailyHealthMetric, k)
    })
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_health_data_for_date(db: Session, target_date) -> Any:
    """Return the health metric row for a specific date, or None."""
    from datetime import date as _date
    from backend.models.health import DailyHealthMetric

    if isinstance(target_date, str):
        target_date = _date.fromisoformat(target_date)
    return db.query(DailyHealthMetric).filter(DailyHealthMetric.date == target_date).first()


def get_recent_health_data(db: Session, days: int = 14) -> List[Any]:
    """Return health metrics for the last `days` days, newest first."""
    from datetime import date as _date, timedelta
    from backend.models.health import DailyHealthMetric

    since = _date.today() - timedelta(days=days)
    return (
        db.query(DailyHealthMetric)
        .filter(DailyHealthMetric.date >= since)
        .order_by(DailyHealthMetric.date.desc())
        .all()
    )
