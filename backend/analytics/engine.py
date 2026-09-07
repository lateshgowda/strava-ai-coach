from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.models.activity import Activity


class AnalyticsEngine:
    """
    Pure pandas/numpy analytics over a list of Activity ORM objects.

    All methods that return DataFrames guarantee the correct columns even when
    there is no data (they return an empty DataFrame with the right schema).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, activities: List[Activity]) -> None:
        self._activities = activities
        self.df = self._build_df()

    def _build_df(self) -> pd.DataFrame:
        """Convert Activity objects to a working DataFrame with derived columns."""
        if not self._activities:
            return pd.DataFrame(columns=[
                "strava_id", "name", "distance", "distance_km", "moving_time",
                "elapsed_time", "elevation_gain", "start_date", "average_heartrate",
                "max_heartrate", "average_cadence", "average_speed", "max_speed",
                "suffer_score", "splits_metric", "activity_type", "pr_count",
                "pace_min_per_km", "week", "month",
            ])

        records = []
        for act in self._activities:
            records.append({
                "strava_id": act.strava_id,
                "name": act.name,
                "distance": act.distance,
                "moving_time": act.moving_time,
                "elapsed_time": act.elapsed_time,
                "elevation_gain": act.elevation_gain,
                "start_date": pd.to_datetime(act.start_date),
                "average_heartrate": act.average_heartrate,
                "max_heartrate": act.max_heartrate,
                "average_cadence": act.average_cadence,
                "average_speed": act.average_speed,
                "max_speed": act.max_speed,
                "suffer_score": act.suffer_score,
                "splits_metric": act.splits_metric,
                "activity_type": act.activity_type,
                "pr_count": act.pr_count,
            })

        df = pd.DataFrame(records)
        df.sort_values("start_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Derived columns
        df["distance_km"] = df["distance"] / 1000.0

        # Pace in min/km — guard against zero speed
        def _pace(row: pd.Series) -> float:
            if row["average_speed"] and row["average_speed"] > 0:
                return (1000.0 / row["average_speed"]) / 60.0
            if row["moving_time"] and row["distance"] and row["distance"] > 0:
                return (row["moving_time"] / row["distance"]) * (1000.0 / 60.0)
            return float("nan")

        df["pace_min_per_km"] = df.apply(_pace, axis=1)

        # ISO week string (e.g., "2024-W15") and month string (e.g., "2024-04")
        df["week"] = df["start_date"].dt.strftime("%G-W%V")
        df["month"] = df["start_date"].dt.strftime("%Y-%m")

        return df

    # ------------------------------------------------------------------
    # Mileage summaries
    # ------------------------------------------------------------------

    def weekly_mileage(self, weeks: int = 12) -> pd.DataFrame:
        """Return per-week aggregate over the last *weeks* ISO weeks."""
        empty = pd.DataFrame(columns=["week", "distance_km", "run_count", "avg_pace", "avg_hr"])
        if self.df.empty:
            return empty

        cutoff = datetime.utcnow() - timedelta(weeks=weeks)
        df = self.df[self.df["start_date"] >= pd.Timestamp(cutoff)].copy()
        if df.empty:
            return empty

        agg = (
            df.groupby("week")
            .agg(
                distance_km=("distance_km", "sum"),
                run_count=("strava_id", "count"),
                avg_pace=("pace_min_per_km", "mean"),
                avg_hr=("average_heartrate", "mean"),
            )
            .reset_index()
            .sort_values("week")
        )
        return agg

    def monthly_mileage(self, months: int = 6) -> pd.DataFrame:
        """Return per-month aggregate over the last *months* calendar months."""
        empty = pd.DataFrame(columns=["month", "distance_km", "run_count", "avg_pace", "avg_hr"])
        if self.df.empty:
            return empty

        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        df = self.df[self.df["start_date"] >= pd.Timestamp(cutoff)].copy()
        if df.empty:
            return empty

        agg = (
            df.groupby("month")
            .agg(
                distance_km=("distance_km", "sum"),
                run_count=("strava_id", "count"),
                avg_pace=("pace_min_per_km", "mean"),
                avg_hr=("average_heartrate", "mean"),
            )
            .reset_index()
            .sort_values("month")
        )
        return agg

    # ------------------------------------------------------------------
    # Trend series
    # ------------------------------------------------------------------

    def pace_trend(self, runs: int = 20) -> pd.DataFrame:
        """Last *runs* activities: date, distance_km, pace_min_per_km."""
        empty = pd.DataFrame(columns=["start_date", "distance_km", "pace_min_per_km", "name"])
        if self.df.empty:
            return empty
        df = self.df.head(runs)[["start_date", "distance_km", "pace_min_per_km", "name"]].copy()
        return df.sort_values("start_date")

    def hr_trend(self, runs: int = 20) -> pd.DataFrame:
        """Last *runs* activities that have HR data."""
        empty = pd.DataFrame(columns=["start_date", "distance_km", "average_heartrate", "name"])
        if self.df.empty:
            return empty
        df = self.df.dropna(subset=["average_heartrate"]).head(runs)[
            ["start_date", "distance_km", "average_heartrate", "name"]
        ].copy()
        return df.sort_values("start_date")

    def cadence_trend(self, runs: int = 20) -> pd.DataFrame:
        """Last *runs* activities that have cadence data."""
        empty = pd.DataFrame(columns=["start_date", "distance_km", "average_cadence", "name"])
        if self.df.empty:
            return empty
        df = self.df.dropna(subset=["average_cadence"]).head(runs)[
            ["start_date", "distance_km", "average_cadence", "name"]
        ].copy()
        return df.sort_values("start_date")

    # ------------------------------------------------------------------
    # Latest run
    # ------------------------------------------------------------------

    def latest_run_stats(self) -> Dict[str, Any]:
        """Return a dict of key stats for the most recent activity."""
        if self.df.empty:
            return {}

        row = self.df.iloc[0]
        stats: Dict[str, Any] = {
            "strava_id": int(row["strava_id"]) if pd.notna(row["strava_id"]) else None,
            "name": row["name"],
            "start_date": str(row["start_date"]),
            "distance_km": round(float(row["distance_km"]), 2),
            "moving_time_sec": int(row["moving_time"]),
            "moving_time_str": _format_duration(int(row["moving_time"])),
            "pace_min_per_km": round(float(row["pace_min_per_km"]), 2) if not math.isnan(row["pace_min_per_km"]) else None,
            "pace_str": _format_pace(float(row["pace_min_per_km"])) if not math.isnan(row["pace_min_per_km"]) else "–",
            "elevation_gain": float(row["elevation_gain"]),
            "average_heartrate": float(row["average_heartrate"]) if pd.notna(row["average_heartrate"]) else None,
            "max_heartrate": float(row["max_heartrate"]) if pd.notna(row["max_heartrate"]) else None,
            "average_cadence": float(row["average_cadence"]) if pd.notna(row["average_cadence"]) else None,
            "pr_count": int(row["pr_count"]) if pd.notna(row["pr_count"]) else 0,
        }

        # Parse splits
        if pd.notna(row["splits_metric"]) and row["splits_metric"]:
            try:
                splits = json.loads(row["splits_metric"])
                stats["splits"] = splits
            except (json.JSONDecodeError, TypeError):
                stats["splits"] = []
        else:
            stats["splits"] = []

        return stats

    # ------------------------------------------------------------------
    # Load / fatigue metrics
    # ------------------------------------------------------------------

    # Load factor per activity type (load units per minute of activity)
    # Run uses distance-based load; others use duration-based load
    _DURATION_LOAD_PER_MIN: Dict[str, float] = {
        "WeightTraining": 0.22,   # ~13 load/hr — comparable to easy run
        "Yoga":           0.06,
        "Pilates":        0.07,
        "Swim":           0.20,
        "Rowing":         0.18,
        "Elliptical":     0.14,
        "StairStepper":   0.16,
        "Crossfit":       0.25,
    }

    def _activity_load(self, row: "pd.Series") -> float:  # noqa: F821
        """Return a load value for a single activity row."""
        atype = row.get("activity_type", "")
        moving_min = (row.get("moving_time") or 0) / 60.0

        if atype == "Run":
            return row["distance_km"] * (1 + (row.get("elevation_gain") or 0) / 1000.0)
        if atype == "Ride":
            return row["distance_km"] * 0.5 * (1 + (row.get("elevation_gain") or 0) / 2000.0)
        if atype == "Walk":
            return row["distance_km"] * 0.3
        if atype == "Hike":
            return row["distance_km"] * 0.6 * (1 + (row.get("elevation_gain") or 0) / 1000.0)

        # Duration-based for everything else
        factor = self._DURATION_LOAD_PER_MIN.get(atype, 0.10)
        return moving_min * factor

    def _weekly_load(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute activity-type-aware Training Stress Score proxy per day.

        - Run / Ride / Walk / Hike: distance + elevation based
        - WeightTraining, Yoga, etc.: duration based with type multiplier
        """
        if df.empty:
            return pd.Series(dtype=float)

        df = df.copy()
        df["_load"] = df.apply(self._activity_load, axis=1)
        daily = df.set_index("start_date").resample("D")["_load"].sum()
        return daily

    def fatigue_score(self) -> float:
        """
        Acute Training Load (ATL) relative to Chronic Training Load (CTL).

        Returns a 0-100 score where higher means more fatigued.
        """
        if self.df.empty:
            return 0.0

        now = pd.Timestamp(datetime.utcnow())
        df28 = self.df[self.df["start_date"] >= now - timedelta(days=28)].copy()
        if df28.empty:
            return 0.0

        load = self._weekly_load(df28)
        atl = float(load[load.index >= now - timedelta(days=7)].sum())
        ctl = float(load.sum()) / 4.0  # 28-day avg as 4-week CTL

        if ctl == 0:
            return 0.0

        ratio = atl / ctl
        # Map ratio to 0-100: ratio=1 -> 50, ratio=2 -> 100
        score = min(100.0, max(0.0, (ratio / 2.0) * 100.0))
        return round(score, 1)

    def recovery_score(self) -> float:
        """
        0-100 score where higher means better recovered.

        Considers fatigue (inverted) and rest days in the last 7 days.
        """
        if self.df.empty:
            return 100.0

        fatigue = self.fatigue_score()
        base = 100.0 - fatigue

        # Bonus for rest days
        now = pd.Timestamp(datetime.utcnow())
        recent = self.df[self.df["start_date"] >= now - timedelta(days=7)]
        run_days = recent["start_date"].dt.date.nunique()
        rest_days = max(0, 7 - run_days)
        rest_bonus = min(15.0, rest_days * 3.0)

        return round(min(100.0, base + rest_bonus), 1)

    def acute_chronic_workload_ratio(self) -> float:
        """
        ACWR = ATL(7d) / CTL(28d).

        Values: <0.8 under-training, 0.8-1.3 optimal, >1.3 injury risk.
        """
        if self.df.empty:
            return 0.0

        now = pd.Timestamp(datetime.utcnow())
        df28 = self.df[self.df["start_date"] >= now - timedelta(days=28)].copy()
        if df28.empty:
            return 0.0

        load = self._weekly_load(df28)
        atl = float(load[load.index >= now - timedelta(days=7)].sum())
        ctl = float(load.sum()) / 4.0

        if ctl == 0:
            return 0.0

        return round(atl / ctl, 2)

    def training_load_trend(self, weeks: int = 8) -> pd.DataFrame:
        """Weekly TSS-proxy scores for the last *weeks* weeks."""
        empty = pd.DataFrame(columns=["week", "load_score", "distance_km"])
        if self.df.empty:
            return empty

        cutoff = datetime.utcnow() - timedelta(weeks=weeks)
        df = self.df[self.df["start_date"] >= pd.Timestamp(cutoff)].copy()
        if df.empty:
            return empty

        df["load_score"] = df.apply(self._activity_load, axis=1)
        agg = (
            df.groupby("week")
            .agg(load_score=("load_score", "sum"), distance_km=("distance_km", "sum"))
            .reset_index()
            .sort_values("week")
        )
        return agg

    def training_status(self) -> Dict[str, Any]:
        """
        Return key load status metrics for the dashboard summary row.

        Covers the rolling 14-day window:
        - consecutive_training_days: current unbroken streak of days with activity
        - this_week_load / last_week_load: rolling 7-day windows
        - load_change_pct: % change between the two windows
        """
        empty: Dict[str, Any] = {
            "consecutive_training_days": 0,
            "this_week_load": 0.0,
            "last_week_load": 0.0,
            "load_change_pct": 0.0,
        }
        if self.df.empty:
            return empty

        now = pd.Timestamp(datetime.utcnow())
        today = now.date()

        # Consecutive training days
        df14 = self.df[self.df["start_date"] >= now - timedelta(days=14)].copy()
        training_dates = set(df14["start_date"].dt.date)
        streak = 0
        for i in range(14):
            if (today - timedelta(days=i)) in training_dates:
                streak += 1
            else:
                break

        # This week vs last week rolling load
        df_this = self.df[self.df["start_date"] >= now - timedelta(days=7)].copy()
        df_last = self.df[
            (self.df["start_date"] >= now - timedelta(days=14))
            & (self.df["start_date"] < now - timedelta(days=7))
        ].copy()

        this_load = float(df_this.apply(self._activity_load, axis=1).sum()) if not df_this.empty else 0.0
        last_load = float(df_last.apply(self._activity_load, axis=1).sum()) if not df_last.empty else 0.0
        change_pct = round(((this_load - last_load) / last_load) * 100, 1) if last_load > 0 else 0.0

        return {
            "consecutive_training_days": streak,
            "this_week_load": round(this_load, 1),
            "last_week_load": round(last_load, 1),
            "load_change_pct": change_pct,
        }

    # ------------------------------------------------------------------
    # Long runs
    # ------------------------------------------------------------------

    def long_run_detection(self) -> pd.DataFrame:
        """Return all runs longer than 15 km."""
        empty = pd.DataFrame(columns=["start_date", "name", "distance_km", "pace_min_per_km", "average_heartrate"])
        if self.df.empty:
            return empty
        df = self.df[self.df["distance_km"] > 15.0][
            ["start_date", "name", "distance_km", "pace_min_per_km", "average_heartrate"]
        ].copy()
        return df.sort_values("start_date")

    def long_run_progression(self) -> pd.DataFrame:
        """Longest run per calendar month."""
        empty = pd.DataFrame(columns=["month", "max_distance_km", "run_name"])
        if self.df.empty:
            return empty

        idx = self.df.groupby("month")["distance_km"].idxmax()
        df = self.df.loc[idx][["month", "distance_km", "name"]].copy()
        df = df.rename(columns={"distance_km": "max_distance_km", "name": "run_name"})
        return df.sort_values("month")

    # ------------------------------------------------------------------
    # Personal records
    # ------------------------------------------------------------------

    def personal_records(self) -> Dict[str, Any]:
        """
        Estimate PRs by finding the fastest equivalent pace over common distances.

        Uses the Riegel formula to project from actual runs.
        """
        if self.df.empty:
            return {"5k": None, "10k": None, "half_marathon": None}

        prs: Dict[str, Any] = {}
        for label, dist_km in [("5k", 5.0), ("10k", 10.0), ("half_marathon", 21.0975)]:
            # Only use runs >= the target distance
            eligible = self.df[self.df["distance_km"] >= dist_km * 0.9].copy()
            if eligible.empty:
                prs[label] = None
                continue
            best_time = eligible.apply(
                lambda row: _riegel_time(
                    actual_dist_km=row["distance_km"],
                    actual_time_sec=row["moving_time"],
                    target_dist_km=dist_km,
                ),
                axis=1,
            ).min()
            if math.isnan(best_time) or math.isinf(best_time):
                prs[label] = None
            else:
                prs[label] = round(best_time)  # seconds
                prs[f"{label}_str"] = _format_duration(int(best_time))

        return prs

    def estimated_race_time(self, distance_km: float = 21.1) -> Optional[float]:
        """
        Estimate a race finish time using the Riegel formula from recent runs.

        Requires at least one run with distance >= distance_km * 0.5.
        Returns estimated time in seconds, or None.
        """
        if self.df.empty:
            return None

        eligible = self.df[self.df["distance_km"] >= distance_km * 0.5].copy()
        if eligible.empty:
            return None

        # Use the 5 most recent eligible runs and take the median estimate
        recent = eligible.head(5)
        estimates = recent.apply(
            lambda row: _riegel_time(
                actual_dist_km=row["distance_km"],
                actual_time_sec=row["moving_time"],
                target_dist_km=distance_km,
            ),
            axis=1,
        )
        estimates = estimates[np.isfinite(estimates)]
        if estimates.empty:
            return None
        return round(float(estimates.median()))

    # ------------------------------------------------------------------
    # Recent trend summary
    # ------------------------------------------------------------------

    def trends_aggregates(self, period: str = "month") -> List[Dict[str, Any]]:
        """
        Return per-period aggregates for all 6 trend metrics.
        period: "week" (last 16 weeks) | "month" (last 12 months) | "year" (all years)
        Each record: label, display, distance_km, avg_pace, activities,
                     elevation_m, moving_time_sec, avg_hr, is_current
        """
        if self.df.empty:
            return []

        df = self.df.copy()

        if period == "week":
            cutoff = datetime.utcnow() - timedelta(weeks=16)
            df = df[df["start_date"] >= pd.Timestamp(cutoff)]
            group_col = "week"
        elif period == "year":
            df = df.copy()
            df["year"] = df["start_date"].dt.year.astype(str)
            group_col = "year"
        else:
            cutoff = datetime.utcnow() - timedelta(days=12 * 30)
            df = df[df["start_date"] >= pd.Timestamp(cutoff)]
            group_col = "month"

        if df.empty:
            return []

        agg = (
            df.groupby(group_col)
            .agg(
                distance_km=("distance_km", "sum"),
                activities=("strava_id", "count"),
                elevation_m=("elevation_gain", "sum"),
                moving_time_sec=("moving_time", "sum"),
                avg_hr=("average_heartrate", "mean"),
                _total_time=("moving_time", "sum"),
                _total_dist=("distance", "sum"),
            )
            .reset_index()
            .sort_values(group_col)
        )

        def _weighted_pace(row: "pd.Series") -> Optional[float]:
            if row["_total_dist"] > 0:
                return (row["_total_time"] / 60.0) / (row["_total_dist"] / 1000.0)
            return None

        agg["avg_pace"] = agg.apply(_weighted_pace, axis=1)

        def _display(key: str) -> str:
            if period == "week":
                try:
                    yr, wk = key.split("-W")
                    monday = datetime.strptime(f"{yr}-{int(wk):02d}-1", "%G-%V-%u")
                    sunday = monday + timedelta(days=6)
                    return sunday.strftime("%-d %b")
                except Exception:
                    return key
            elif period == "month":
                try:
                    y, m = key.split("-")
                    return datetime(int(y), int(m), 1).strftime("%b %y")
                except Exception:
                    return key
            return key

        agg["display"] = agg[group_col].apply(_display)

        now = datetime.utcnow()
        if period == "week":
            current_key = now.strftime("%G-W%V")
        elif period == "year":
            current_key = str(now.year)
        else:
            current_key = now.strftime("%Y-%m")

        agg["is_current"] = agg[group_col] == current_key
        agg.rename(columns={group_col: "label"}, inplace=True)

        keep = ["label", "display", "distance_km", "avg_pace", "activities",
                "elevation_m", "moving_time_sec", "avg_hr", "is_current"]
        return agg[[c for c in keep if c in agg.columns]].to_dict(orient="records")

    def recent_trend_summary(self) -> Dict[str, Any]:
        """Return a compact dict comparing last 4 weeks vs previous 4 weeks."""
        default = {
            "avg_weekly_km_last4w": 0.0,
            "avg_weekly_km_prev4w": 0.0,
            "trend_pct": 0.0,
            "avg_pace_last4w": None,
            "avg_hr_last4w": None,
        }
        if self.df.empty:
            return default

        now = pd.Timestamp(datetime.utcnow())
        last4w = self.df[self.df["start_date"] >= now - timedelta(days=28)]
        prev4w = self.df[
            (self.df["start_date"] >= now - timedelta(days=56))
            & (self.df["start_date"] < now - timedelta(days=28))
        ]

        avg_last = float(last4w["distance_km"].sum()) / 4.0
        avg_prev = float(prev4w["distance_km"].sum()) / 4.0

        trend_pct = 0.0
        if avg_prev > 0:
            trend_pct = round(((avg_last - avg_prev) / avg_prev) * 100.0, 1)

        avg_pace = (
            float(last4w["pace_min_per_km"].dropna().mean())
            if not last4w["pace_min_per_km"].dropna().empty
            else None
        )
        avg_hr = (
            float(last4w["average_heartrate"].dropna().mean())
            if not last4w["average_heartrate"].dropna().empty
            else None
        )

        return {
            "avg_weekly_km_last4w": round(avg_last, 1),
            "avg_weekly_km_prev4w": round(avg_prev, 1),
            "trend_pct": trend_pct,
            "avg_pace_last4w": round(avg_pace, 2) if avg_pace else None,
            "avg_hr_last4w": round(avg_hr, 1) if avg_hr else None,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: int) -> str:
    """Convert seconds to H:MM:SS or M:SS string."""
    if seconds <= 0:
        return "0:00"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_pace(pace_min_per_km: float) -> str:
    """Convert decimal min/km to MM:SS/km string."""
    if math.isnan(pace_min_per_km) or pace_min_per_km <= 0:
        return "–"
    minutes = int(pace_min_per_km)
    seconds = int(round((pace_min_per_km - minutes) * 60))
    return f"{minutes}:{seconds:02d} /km"


def _riegel_time(
    actual_dist_km: float,
    actual_time_sec: float,
    target_dist_km: float,
    exponent: float = 1.06,
) -> float:
    """
    Riegel's endurance formula: t2 = t1 * (d2/d1)^1.06

    Returns predicted time in seconds.
    """
    if actual_dist_km <= 0 or actual_time_sec <= 0:
        return float("inf")
    return actual_time_sec * ((target_dist_km / actual_dist_km) ** exponent)
