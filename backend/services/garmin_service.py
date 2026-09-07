"""Garmin Connect integration — auth, daily metrics sync, workout push."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.garmin import GarminDailyMetrics

_MFA_STATE_FILE = Path("/data/garmin_mfa_state.json")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session type → HR zone mapping
# ---------------------------------------------------------------------------

_ZONE_MAP = {
    "recovery": 1,
    "easy": 2,
    "base": 2,
    "long": 2,
    "aerobic": 2,
    "moderate": 3,
    "steady": 3,
    "tempo": 4,
    "threshold": 4,
    "marathon": 4,
    "interval": 5,
    "speed": 5,
    "race": 5,
}


def _session_zone(session_type: str) -> int:
    st_lower = (session_type or "run").lower()
    for key, zone in _ZONE_MAP.items():
        if key in st_lower:
            return zone
    return 2


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def connect(email: str, password: str, mfa_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Authenticate with Garmin Connect.
    Returns {"status": "connected", "tokenstore": <str>}
              {"status": "mfa_required"}
    Raises HTTPException 401 on wrong credentials.
    """
    from fastapi import HTTPException
    from garminconnect import Garmin, GarminConnectAuthenticationError

    try:
        if mfa_code:
            # Resume a pending MFA login
            if not _MFA_STATE_FILE.exists():
                raise HTTPException(status_code=400, detail="MFA session expired — please reconnect.")
            state = json.loads(_MFA_STATE_FILE.read_text())
            api = Garmin(state["email"], state["password"], return_on_mfa=True)
            api.resume_login(state["client_state"], mfa_code)
            _MFA_STATE_FILE.unlink(missing_ok=True)
        else:
            api = Garmin(email, password, return_on_mfa=True)
            mfa_status, client_state = api.login()
            if mfa_status == "needs_mfa":
                _MFA_STATE_FILE.write_text(json.dumps({
                    "email": email,
                    "password": password,
                    "client_state": client_state,
                }))
                return {"status": "mfa_required"}

        tokenstore = api.client.dumps()
        return {"status": "connected", "tokenstore": tokenstore}

    except GarminConnectAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=f"Garmin authentication failed: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Garmin connection error: {exc}")


def get_client(tokenstore: str):
    """Restore a Garmin client from saved tokenstore."""
    from garminconnect import Garmin

    api = Garmin()
    api.login(tokenstore=tokenstore)
    return api


# ---------------------------------------------------------------------------
# Daily metrics sync
# ---------------------------------------------------------------------------


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception as exc:
        logger.debug("Garmin fetch error: %s", exc)
        return default


def _parse_body_battery(data: Any) -> tuple[Optional[int], Optional[int]]:
    if not data or not isinstance(data, list):
        return None, None
    try:
        charged_vals = [d.get("charged") for d in data if isinstance(d, dict) and d.get("charged") is not None]
        drained_vals = [d.get("drained") for d in data if isinstance(d, dict) and d.get("drained") is not None]
        bb_max = max(charged_vals) if charged_vals else None
        # min = lowest charged minus total drained
        if charged_vals and drained_vals:
            bb_min = max(0, min(charged_vals) - sum(drained_vals))
        else:
            bb_min = None
        return bb_max, bb_min
    except Exception:
        return None, None


def _parse_sleep(data: Any) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "sleep_duration_sec": None,
        "sleep_deep_sec": None,
        "sleep_rem_sec": None,
        "sleep_light_sec": None,
        "sleep_awake_sec": None,
        "sleep_score": None,
    }
    if not data or not isinstance(data, dict):
        return out
    try:
        dto = data.get("dailySleepDTO", {}) or {}
        deep = dto.get("deepSleepSeconds")
        light = dto.get("lightSleepSeconds")
        rem = dto.get("remSleepSeconds")
        awake = dto.get("awakeSleepSeconds")
        if deep is not None:
            out["sleep_deep_sec"] = int(deep)
        if light is not None:
            out["sleep_light_sec"] = int(light)
        if rem is not None:
            out["sleep_rem_sec"] = int(rem)
        if awake is not None:
            out["sleep_awake_sec"] = int(awake)
        parts = [v for v in [deep, light, rem] if v is not None]
        if parts:
            out["sleep_duration_sec"] = int(sum(parts))
        # Sleep score
        scores = dto.get("sleepScores") or data.get("sleepScores")
        if isinstance(scores, dict):
            overall = scores.get("overall")
            if isinstance(overall, dict):
                out["sleep_score"] = overall.get("value")
            elif isinstance(overall, (int, float)):
                out["sleep_score"] = int(overall)
    except Exception:
        pass
    return out


def _parse_rhr(data: Any) -> Optional[int]:
    if not data:
        return None
    try:
        metrics_map = data.get("allMetrics", {}).get("metricsMap", {})
        rhr_list = metrics_map.get("WELLNESS_RESTING_HEART_RATE", [])
        if rhr_list:
            val = rhr_list[0].get("value")
            return int(val) if val is not None else None
    except Exception:
        pass
    return None


def _parse_stress(data: Any) -> Optional[int]:
    if not data:
        return None
    try:
        values = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    v = item[1]
                    if isinstance(v, (int, float)) and v > 0:
                        values.append(v)
                elif isinstance(item, dict):
                    v = item.get("stressLevel") or item.get("value")
                    if isinstance(v, (int, float)) and v > 0:
                        values.append(v)
        return int(sum(values) / len(values)) if values else None
    except Exception:
        return None


def _parse_vo2max(data: Any) -> Optional[float]:
    if not data:
        return None
    try:
        # Response is a list: [{"generic": {"vo2MaxPreciseValue": 42.6, "vo2MaxValue": 43.0}, ...}]
        if isinstance(data, list) and data:
            generic = data[0].get("generic") or {}
            for key in ("vo2MaxPreciseValue", "vo2MaxValue"):
                val = generic.get(key)
                if val is not None:
                    return float(val)
        # Fallback: dict with allMetrics
        if isinstance(data, dict):
            metrics_map = data.get("allMetrics", {}).get("metricsMap", {})
            for key in ("MAX_MET_SPEED_LAST_7_DAYS", "VO2_MAX_VALUE", "vo2MaxValue"):
                entries = metrics_map.get(key)
                if entries and isinstance(entries, list):
                    val = entries[0].get("value")
                    if val is not None:
                        return float(val)
            for key in ("vo2MaxValue", "vo2Max"):
                val = data.get(key)
                if val is not None:
                    return float(val)
    except Exception:
        pass
    return None


def _parse_recovery_time(data: Any) -> Optional[int]:
    if not data:
        return None
    try:
        if isinstance(data, dict):
            for key in ("recoveryTime", "recoveryTimeInHours", "recoveryTimeSeconds"):
                val = data.get(key)
                if val is not None:
                    hours = int(val) // 3600 if key == "recoveryTimeSeconds" else int(val)
                    return hours
            # Nested
            for sub_key in ("trainingStatusDTO", "trainingStatus"):
                sub = data.get(sub_key)
                if isinstance(sub, dict):
                    val = sub.get("recoveryTime")
                    if val is not None:
                        return int(val)
    except Exception:
        pass
    return None


def sync_day(db: Session, api, date_str: str) -> GarminDailyMetrics:
    """Fetch all metrics for one day and upsert into DB."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()

    bb_data = _safe(api.get_body_battery, date_str, default=None)
    bb_max, bb_min = _parse_body_battery(bb_data)

    sleep_data = _safe(api.get_sleep_data, date_str, default=None)
    sleep = _parse_sleep(sleep_data)

    rhr_data = _safe(api.get_rhr_day, date_str, default=None)
    resting_hr = _parse_rhr(rhr_data)

    stress_data = _safe(api.get_stress_data, date_str, default=None)
    stress_avg = _parse_stress(stress_data)

    vo2_data = _safe(api.get_max_metrics, date_str, default=None)
    vo2max = _parse_vo2max(vo2_data)

    ts_data = _safe(api.get_training_status, date_str, default=None)
    recovery_time_hours = _parse_recovery_time(ts_data)

    existing = db.query(GarminDailyMetrics).filter(GarminDailyMetrics.date == d).first()
    if existing:
        row = existing
    else:
        row = GarminDailyMetrics(date=d)
        db.add(row)

    row.body_battery_max = bb_max
    row.body_battery_min = bb_min
    row.stress_avg = stress_avg
    row.resting_hr = resting_hr
    row.sleep_duration_sec = sleep["sleep_duration_sec"]
    row.sleep_deep_sec = sleep["sleep_deep_sec"]
    row.sleep_rem_sec = sleep["sleep_rem_sec"]
    row.sleep_light_sec = sleep["sleep_light_sec"]
    row.sleep_awake_sec = sleep["sleep_awake_sec"]
    row.sleep_score = sleep["sleep_score"]
    row.recovery_time_hours = recovery_time_hours
    row.vo2max = vo2max
    row.synced_at = datetime.utcnow()
    db.commit()
    return row


def sync_recent(db: Session, tokenstore: str, days: int = 7) -> Dict[str, Any]:
    """Sync the last N days of Garmin metrics."""
    try:
        api = get_client(tokenstore)
    except Exception as exc:
        return {"synced": 0, "errors": [f"Auth failed: {exc}"]}

    synced = 0
    errors: List[str] = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        try:
            sync_day(db, api, date_str)
            synced += 1
        except Exception as exc:
            errors.append(f"{date_str}: {exc}")
            logger.warning("Garmin sync error for %s: %s", date_str, exc)

    return {"synced": synced, "errors": errors}


# ---------------------------------------------------------------------------
# Workout builder + push
# ---------------------------------------------------------------------------


def _make_step(order: int, type_id: int, type_key: str, dist_m: float,
               target_zone: Optional[int]) -> Dict[str, Any]:
    if target_zone:
        target = {
            "workoutTargetTypeId": 4,
            "workoutTargetTypeKey": "heart.rate.zone",
        }
        tv1 = tv2 = target_zone
    else:
        target = {
            "workoutTargetTypeId": 1,
            "workoutTargetTypeKey": "no.target",
        }
        tv1 = tv2 = None

    return {
        "type": "ExecutableStepDTO",
        "stepOrder": order,
        "stepType": {"stepTypeId": type_id, "stepTypeKey": type_key},
        "endCondition": {"conditionTypeId": 3, "conditionTypeKey": "distance"},
        "endConditionValue": float(dist_m),
        "targetType": target,
        "targetValueOne": tv1,
        "targetValueTwo": tv2,
    }


def build_workout(plan_workout: Dict[str, Any]) -> Dict[str, Any]:
    dist_km: float = plan_workout.get("distance_km", 5.0)
    session_type: str = plan_workout.get("session_type", "Run")
    details: str = plan_workout.get("details") or ""
    zone = _session_zone(session_type)
    dist_m = dist_km * 1000.0

    steps: List[Dict[str, Any]] = []
    if dist_km >= 6.0:
        steps.append(_make_step(1, 1, "warmup", 1000.0, None))
        steps.append(_make_step(2, 3, "interval", dist_m - 2000.0, zone))
        steps.append(_make_step(3, 2, "cooldown", 1000.0, None))
    elif dist_km >= 3.0:
        steps.append(_make_step(1, 3, "interval", dist_m, zone))
    else:
        steps.append(_make_step(1, 3, "interval", dist_m, None))

    return {
        "workoutName": f"{session_type} — {dist_km:.1f}km",
        "description": details,
        "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
        "workoutSegments": [{
            "segmentOrder": 1,
            "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
            "workoutSteps": steps,
        }],
    }


def push_workout(api, plan_workout: Dict[str, Any]) -> Dict[str, Any]:
    workout_json = build_workout(plan_workout)
    result = api.upload_workout(workout_json)
    workout_id = result["workoutId"]
    plan_date = plan_workout["plan_date"]
    api.schedule_workout(workout_id, plan_date)
    return {"workout_id": workout_id, "scheduled": plan_date, "name": workout_json["workoutName"]}
