from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.ai.coach import generate_run_insight, generate_weekly_summary
from backend.analytics.engine import AnalyticsEngine, _format_duration, _format_pace
from backend.db.crud import get_activities, get_health_data_for_date, get_latest_activity, get_token
from backend.strava.sync import sync_activities


class ActivityService:
    """
    Orchestration layer between the HTTP API and the domain logic.

    All heavy lifting (sync, analytics, AI) is delegated to the relevant
    modules; this class only assembles results.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, full_sync: bool = False) -> Dict[str, Any]:
        """Trigger a Strava sync. Returns {"synced": n, "skipped": n, "total": n}."""
        return sync_activities(self._db, full_sync=full_sync)

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Assemble all analytics data for the frontend dashboard.

        DataFrames are serialised to list[dict] so they are JSON-serialisable.
        """
        token = get_token(self._db)
        is_connected = token is not None

        # Running activities — for pace, HR, cadence, long runs, PRs, HM estimate
        run_activities = get_activities(self._db, limit=500, activity_type="Run")
        # All activities — for fatigue, recovery, ACWR (reflects full training load)
        all_activities = get_activities(self._db, limit=2000, activity_type=None)

        run_engine = AnalyticsEngine(run_activities)
        all_engine = AnalyticsEngine(all_activities)

        # --- Running-specific analytics ---
        weekly_df = run_engine.weekly_mileage(weeks=12)
        monthly_df = run_engine.monthly_mileage(months=6)
        pace_trend_df = run_engine.pace_trend(runs=20)
        hr_trend_df = run_engine.hr_trend(runs=20)
        cadence_trend_df = run_engine.cadence_trend(runs=20)
        long_runs_df = run_engine.long_run_progression()
        prs = run_engine.personal_records()
        est_hm = run_engine.estimated_race_time(distance_km=21.1)
        latest = run_engine.latest_run_stats()
        recent_trend = run_engine.recent_trend_summary()

        # --- Load metrics: all activity types (runs + strength + rides + walks + yoga) ---
        fatigue = all_engine.fatigue_score()
        recovery = all_engine.recovery_score()
        acwr = all_engine.acute_chronic_workload_ratio()
        load_trend_df = all_engine.training_load_trend(weeks=8)

        # --- Activity type breakdown for AI context ---
        from collections import Counter
        type_counts = Counter(a.activity_type for a in all_activities)

        # --- Training status (streak, week-over-week load) ---
        training_status = all_engine.training_status()

        # --- Advanced training state (deterministic analytics) ---
        from backend.analytics.training_state import TrainingStateEngine
        from backend.analytics.readiness import compute_readiness
        from backend.analytics.injury_risk import compute_injury_risk
        from backend.analytics.recommendations import generate_weekly_plan
        from backend.analytics.race_predictor import compute_race_predictions

        ts_engine = TrainingStateEngine(all_activities, run_activities=run_activities)
        training_state = ts_engine.compute_all()

        # Fetch today's Apple Health data (synced by iOS Shortcut).
        # Fall back to yesterday if today's sync hasn't run yet (e.g. before 9 AM).
        import datetime as _datetime_mod
        today_health = get_health_data_for_date(self._db, _datetime_mod.date.today())
        if today_health is None:
            yesterday = _datetime_mod.date.today() - _datetime_mod.timedelta(days=1)
            today_health = get_health_data_for_date(self._db, yesterday)

        # Fetch today's Garmin daily metrics (Body Battery, stress, recovery time, VO2 max).
        # Also used as sleep/RHR fallback when Apple Health is not synced.
        garmin_today = None
        _garmin_vo2max = None
        try:
            from backend.models.garmin import GarminDailyMetrics as _GDM
            _garmin_row = (
                self._db.query(_GDM)
                .filter(_GDM.date == _datetime_mod.date.today())
                .first()
            )
            if _garmin_row is None:
                _gd_yest = _datetime_mod.date.today() - _datetime_mod.timedelta(days=1)
                _garmin_row = (
                    self._db.query(_GDM)
                    .filter(_GDM.date == _gd_yest)
                    .first()
                )
            garmin_today = _garmin_row
            # Find most recent VO2 max (only synced occasionally)
            _vo2_row = (
                self._db.query(_GDM)
                .filter(_GDM.vo2max.isnot(None))
                .order_by(_GDM.date.desc())
                .first()
            )
            if _vo2_row:
                _garmin_vo2max = _vo2_row.vo2max
        except Exception:
            pass

        # Resolve sleep / RHR: Apple Health first, Garmin fallback
        _sleep_hours = today_health.sleep_duration_hours if today_health else None
        _sleep_deep_hours = today_health.sleep_deep_hours if today_health else None
        _resting_hr = today_health.resting_hr if today_health else None
        if garmin_today:
            if _sleep_hours is None and garmin_today.sleep_duration_sec:
                _sleep_hours = garmin_today.sleep_duration_sec / 3600.0
            if _sleep_deep_hours is None and garmin_today.sleep_deep_sec:
                _sleep_deep_hours = garmin_today.sleep_deep_sec / 3600.0
            if _resting_hr is None and garmin_today.resting_hr:
                _resting_hr = garmin_today.resting_hr

        # Pull profile baseline resting HR for elevation comparison
        _baseline_rhr = None
        try:
            from backend.db.crud import get_profile as _get_profile
            _profile = _get_profile(self._db)
            if _profile and _profile.resting_hr:
                _baseline_rhr = _profile.resting_hr
        except Exception:
            pass

        readiness = compute_readiness(
            fatigue_score=fatigue,
            recovery_score=recovery,
            acwr=acwr,
            monotony_score=training_state["monotony_score"],
            consecutive_training_days=training_status["consecutive_training_days"],
            recovery_debt=training_state["recovery_debt"],
            overreaching=training_state["overreaching"],
            sleep_hours=_sleep_hours,
            sleep_deep_hours=_sleep_deep_hours,
            resting_hr=_resting_hr,
            baseline_resting_hr=_baseline_rhr,
            body_battery_max=garmin_today.body_battery_max if garmin_today else None,
            recovery_time_hours=garmin_today.recovery_time_hours if garmin_today else None,
            stress_avg=garmin_today.stress_avg if garmin_today else None,
        )

        injury_risk = compute_injury_risk(
            acwr=acwr,
            monotony_score=training_state["monotony_score"],
            consecutive_training_days=training_status["consecutive_training_days"],
            load_change_pct=training_status["load_change_pct"],
            recovery_debt=training_state["recovery_debt"],
            recovery_score=recovery,
            fatigue_score=fatigue,
        )

        race_predictions = compute_race_predictions(
            personal_records=prs,
            consistency_score=training_state["consistency_score"],
            aerobic_fitness_score=training_state["aerobic_fitness_score"],
            fatigue_score=fatigue,
            hr_efficiency_trend=training_state["hr_efficiency_trend"]["trend"],
            pace_efficiency_trend=training_state["pace_efficiency_trend"]["trend"],
            acwr=acwr,
            vo2max=_garmin_vo2max,
        )

        weekly_plan = generate_weekly_plan(
            acwr=acwr,
            fatigue_score=fatigue,
            recovery_score=recovery,
            readiness_score=readiness["readiness_score"],
            this_week_km=recent_trend.get("avg_weekly_km_last4w", 0.0),
            avg_weekly_km_last4w=recent_trend.get("avg_weekly_km_last4w", 20.0),
            consistency_score=training_state["consistency_score"],
            consecutive_training_days=training_status["consecutive_training_days"],
        )

        # Load athlete profile for context (if set)
        try:
            from backend.db.crud import get_profile
            profile = get_profile(self._db)
            if profile and profile.primary_goal:
                weekly_plan = generate_weekly_plan(
                    acwr=acwr,
                    fatigue_score=fatigue,
                    recovery_score=recovery,
                    readiness_score=readiness["readiness_score"],
                    this_week_km=recent_trend.get("avg_weekly_km_last4w", 0.0),
                    avg_weekly_km_last4w=recent_trend.get("avg_weekly_km_last4w", 20.0),
                    consistency_score=training_state["consistency_score"],
                    consecutive_training_days=training_status["consecutive_training_days"],
                    preferred_weekly_km=profile.preferred_weekly_km,
                    primary_goal=profile.primary_goal,
                    target_race_date_days=(
                        (profile.target_race_date - __import__("datetime").datetime.utcnow()).days
                        if profile.target_race_date else None
                    ),
                    weekly_training_hours=profile.weekly_training_hours,
                    running_experience=profile.running_experience,
                    max_hr=profile.max_hr,
                )
        except Exception:
            pass  # profile not available — use defaults

        return {
            "latest_run": latest,
            "weekly": _df_to_records(weekly_df),
            "monthly": _df_to_records(monthly_df),
            "pace_trend": _df_to_records(pace_trend_df),
            "hr_trend": _df_to_records(hr_trend_df),
            "cadence_trend": _df_to_records(cadence_trend_df),
            "fatigue_score": fatigue,
            "recovery_score": recovery,
            "acwr": acwr,
            "training_load_trend": _df_to_records(load_trend_df),
            "long_runs": _df_to_records(long_runs_df),
            "personal_records": prs,
            "estimated_hm_time": est_hm,
            "estimated_hm_str": _format_duration(int(est_hm)) if est_hm else None,
            "recent_trend": recent_trend,
            "total_activities": len(all_activities),
            "total_runs": len(run_activities),
            "activity_type_counts": dict(type_counts),
            "training_status": training_status,
            "training_state": training_state,
            "readiness": readiness,
            "injury_risk": injury_risk,
            "race_predictions": race_predictions,
            "weekly_plan": weekly_plan,
            "is_connected": is_connected,
            "athlete_name": token.athlete_name if token else None,
            "today_health": _health_row_to_dict(today_health) if today_health else None,
            "garmin_today": {**_garmin_row_to_dict(garmin_today), "vo2max_latest": _garmin_vo2max} if garmin_today else None,
        }

    # ------------------------------------------------------------------
    # AI insights
    # ------------------------------------------------------------------

    def get_ai_insight(self) -> str:
        """Build a concise metrics dict and request a coaching insight."""
        activities = get_activities(self._db, limit=200, activity_type="Run")
        engine = AnalyticsEngine(activities)

        latest = engine.latest_run_stats()
        recent_trend = engine.recent_trend_summary()
        prs = engine.personal_records()
        est_hm = engine.estimated_race_time(distance_km=21.1)

        metrics: Dict[str, Any] = {
            "latest_run": {
                "name": latest.get("name", "Unknown"),
                "distance_km": latest.get("distance_km"),
                "pace": latest.get("pace_str"),
                "time": latest.get("moving_time_str"),
                "average_hr": latest.get("average_heartrate"),
                "cadence": latest.get("average_cadence"),
                "elevation_gain_m": latest.get("elevation_gain"),
            },
            "training_load": {
                "fatigue_score_0_100": engine.fatigue_score(),
                "recovery_score_0_100": engine.recovery_score(),
                "acwr": engine.acute_chronic_workload_ratio(),
            },
            "trends_last_4_weeks": {
                "avg_weekly_km": recent_trend.get("avg_weekly_km_last4w"),
                "prev_4w_avg_weekly_km": recent_trend.get("avg_weekly_km_prev4w"),
                "volume_change_pct": recent_trend.get("trend_pct"),
                "avg_pace_min_per_km": recent_trend.get("avg_pace_last4w"),
                "avg_hr": recent_trend.get("avg_hr_last4w"),
            },
            "personal_records": {
                "5k": prs.get("5k_str"),
                "10k": prs.get("10k_str"),
                "half_marathon": prs.get("half_marathon_str"),
            },
            "estimated_half_marathon": _format_duration(int(est_hm)) if est_hm else "N/A",
        }

        return generate_run_insight(metrics)

    def get_weekly_ai_summary(self) -> str:
        """Build weekly aggregates and request a weekly summary."""
        activities = get_activities(self._db, limit=200, activity_type="Run")
        engine = AnalyticsEngine(activities)

        weekly_df = engine.weekly_mileage(weeks=4)
        recent_trend = engine.recent_trend_summary()

        weekly_records = _df_to_records(weekly_df)

        # Build a clean summary
        current_week = weekly_records[-1] if weekly_records else {}
        prev_weeks = weekly_records[:-1] if len(weekly_records) > 1 else []

        weekly_data: Dict[str, Any] = {
            "current_week": {
                "week": current_week.get("week", "N/A"),
                "distance_km": current_week.get("distance_km", 0),
                "run_count": current_week.get("run_count", 0),
                "avg_pace_min_per_km": current_week.get("avg_pace"),
                "avg_hr": current_week.get("avg_hr"),
            },
            "previous_weeks_summary": {
                "avg_weekly_km": recent_trend.get("avg_weekly_km_prev4w"),
                "weeks_included": len(prev_weeks),
            },
            "training_load": {
                "fatigue_score_0_100": engine.fatigue_score(),
                "recovery_score_0_100": engine.recovery_score(),
                "acwr": engine.acute_chronic_workload_ratio(),
            },
            "trend": {
                "volume_change_pct": recent_trend.get("trend_pct"),
                "avg_weekly_km_last_4w": recent_trend.get("avg_weekly_km_last4w"),
            },
        }

        return generate_weekly_summary(weekly_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_to_records(df) -> List[Dict[str, Any]]:
    """Serialise a pandas DataFrame to a list of plain dicts."""
    if df is None or df.empty:
        return []
    # Convert datetime columns to ISO strings
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    # Replace NaN/inf with None for JSON safety
    import math

    def _clean(val):
        if val is None:
            return None
        try:
            if math.isnan(val) or math.isinf(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    records = df.to_dict(orient="records")
    return [{k: _clean(v) for k, v in rec.items()} for rec in records]


def _health_row_to_dict(row: Any) -> Dict[str, Any]:
    """Serialise a DailyHealthMetric row, deriving total sleep from stages if stored total is missing."""
    total = row.sleep_duration_hours
    if not total:
        stage_sum = sum(
            v for v in (row.sleep_deep_hours, row.sleep_rem_hours, row.sleep_core_hours)
            if v is not None
        )
        total = round(stage_sum, 2) if stage_sum > 0 else None
    return {
        "date": row.date.isoformat() if row.date else None,
        "sleep_duration_hours": total,
        "sleep_deep_hours": row.sleep_deep_hours,
        "sleep_rem_hours": row.sleep_rem_hours,
        "sleep_core_hours": row.sleep_core_hours,
        "sleep_awake_hours": row.sleep_awake_hours,
        "resting_hr": row.resting_hr,
    }


def _garmin_row_to_dict(row: Any) -> Dict[str, Any]:
    """Serialise a GarminDailyMetrics row."""
    return {
        "date": row.date.isoformat() if row.date else None,
        "body_battery_max": row.body_battery_max,
        "body_battery_min": row.body_battery_min,
        "stress_avg": row.stress_avg,
        "resting_hr": row.resting_hr,
        "sleep_duration_hours": row.sleep_duration_sec / 3600.0 if row.sleep_duration_sec else None,
        "sleep_deep_hours": row.sleep_deep_sec / 3600.0 if row.sleep_deep_sec else None,
        "sleep_rem_hours": row.sleep_rem_sec / 3600.0 if row.sleep_rem_sec else None,
        "sleep_light_hours": row.sleep_light_sec / 3600.0 if row.sleep_light_sec else None,
        "sleep_score": row.sleep_score,
        "recovery_time_hours": row.recovery_time_hours,
        "vo2max": row.vo2max,
    }
