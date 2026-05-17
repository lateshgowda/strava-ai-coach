import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.db.crud import (
    delete_all_activities,
    get_latest_activity,
    upsert_activity,
)
from backend.strava.auth import get_valid_token
from backend.strava.client import StravaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Strip timezone info to store as naive UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    return None


def parse_activity(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the fields we persist from a raw Strava activity dict.

    Handles both summary (list endpoint) and detail (single-activity endpoint)
    payloads — splits_metric is only present in detail payloads.
    """
    splits_raw = raw.get("splits_metric")
    splits_json: Optional[str] = None
    if splits_raw and isinstance(splits_raw, list):
        try:
            splits_json = json.dumps(splits_raw)
        except (TypeError, ValueError):
            splits_json = None

    start_date = _parse_date(raw.get("start_date"))
    if start_date is None:
        start_date = datetime.utcnow()

    return {
        "strava_id": int(raw["id"]),
        "name": raw.get("name", ""),
        "distance": _safe_float(raw.get("distance")) or 0.0,
        "moving_time": _safe_int(raw.get("moving_time")) or 0,
        "elapsed_time": _safe_int(raw.get("elapsed_time")) or 0,
        "elevation_gain": _safe_float(raw.get("total_elevation_gain")) or 0.0,
        "start_date": start_date,
        "average_heartrate": _safe_float(raw.get("average_heartrate")),
        "max_heartrate": _safe_float(raw.get("max_heartrate")),
        "average_cadence": _safe_float(raw.get("average_cadence")),
        "average_speed": _safe_float(raw.get("average_speed")) or 0.0,
        "max_speed": _safe_float(raw.get("max_speed")),
        "suffer_score": _safe_int(raw.get("suffer_score")),
        "splits_metric": splits_json,
        "activity_type": raw.get("type", "Run"),
        "pr_count": _safe_int(raw.get("pr_count")),
    }


# ---------------------------------------------------------------------------
# Main sync function
# ---------------------------------------------------------------------------


def sync_activities(db: Session, full_sync: bool = False) -> Dict[str, int]:
    """
    Pull activities from Strava and persist them to the database.

    Parameters
    ----------
    full_sync: When True, wipe existing data and re-fetch everything.
               When False, only fetch runs newer than the most recent stored activity.

    Returns
    -------
    {"synced": n, "skipped": n, "total": n}
    """
    access_token = get_valid_token(db)
    if not access_token:
        raise RuntimeError("No valid Strava token available. Please connect your Strava account.")

    client = StravaClient(access_token)

    after_ts: Optional[int] = None

    if full_sync:
        logger.info("Full sync requested — deleting all existing activities.")
        delete_all_activities(db)
    else:
        latest = get_latest_activity(db)
        if latest:
            # Add one second to avoid re-fetching the latest activity
            after_ts = int(latest.start_date.replace(tzinfo=timezone.utc).timestamp()) + 1
            logger.info("Incremental sync — fetching activities after %s", latest.start_date)

    logger.info("Fetching activities from Strava (after=%s, full=%s)…", after_ts, full_sync)
    raw_activities = client.get_all_activities(after=after_ts)

    logger.info("Fetched %d total activities.", len(raw_activities))

    synced = 0
    skipped = 0

    # Sync all activity types (Run, Ride, WeightTraining, Walk, etc.)
    # Use summary data only — no per-activity detail calls to avoid rate limits.
    for summary in raw_activities:
        activity_data = parse_activity(summary)
        _, is_new = upsert_activity(db, activity_data)
        if is_new:
            synced += 1
        else:
            skipped += 1

    total = synced + skipped
    logger.info("Sync complete: %d new, %d updated, %d total.", synced, skipped, total)
    return {"synced": synced, "skipped": skipped, "total": total}
