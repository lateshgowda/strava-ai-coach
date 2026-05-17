"""
Advanced training state analytics.

Based on Banister impulse-response model principles and published sports
science literature.  All calculations are deterministic — no LLM.

References:
- Banister et al. (1975) — ATL/CTL model
- Gabbett (2016) — ACWR and injury risk
- Coggan (2003) — Training Stress Score principles
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.models.activity import Activity


class TrainingStateEngine:
    """
    Computes advanced training state metrics over a list of Activity records.

    Uses two DataFrames:
      df_all  — all activity types (for load / fatigue metrics)
      df_runs — runs only (for pace / HR efficiency / long-run analysis)
    """

    # Activity-type load factors (mirrors AnalyticsEngine for consistency)
    _DURATION_LOAD_PER_MIN: Dict[str, float] = {
        "WeightTraining": 0.22,
        "Yoga": 0.06,
        "Pilates": 0.07,
        "Swim": 0.20,
        "Rowing": 0.18,
        "Elliptical": 0.14,
        "StairStepper": 0.16,
        "Crossfit": 0.25,
    }

    def __init__(
        self,
        activities: List[Activity],
        run_activities: Optional[List[Activity]] = None,
    ) -> None:
        self._all = activities
        self._runs = run_activities or [a for a in activities if getattr(a, "activity_type", "") == "Run"]
        self.df_all = self._build_df(self._all)
        self.df_runs = self._build_df(self._runs)

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _build_df(self, activities: List[Activity]) -> pd.DataFrame:
        if not activities:
            return pd.DataFrame()
        records = []
        for a in activities:
            records.append({
                "strava_id": a.strava_id,
                "name": a.name or "",
                "distance_km": (a.distance or 0.0) / 1000.0,
                "moving_time": a.moving_time or 0,
                "elevation_gain": a.elevation_gain or 0.0,
                "start_date": pd.to_datetime(a.start_date),
                "average_heartrate": a.average_heartrate,
                "average_cadence": a.average_cadence,
                "average_speed": a.average_speed or 0.0,
                "activity_type": a.activity_type or "Run",
                "splits_metric": a.splits_metric,
            })
        df = pd.DataFrame(records)
        df.sort_values("start_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Derived pace column
        def _pace(row: pd.Series) -> float:
            if row["average_speed"] > 0:
                return (1000.0 / row["average_speed"]) / 60.0
            return float("nan")

        df["pace_min_per_km"] = df.apply(_pace, axis=1)
        df["week"] = df["start_date"].dt.strftime("%G-W%V")
        df["date"] = df["start_date"].dt.date
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activity_load(self, row: Any) -> float:
        """Activity-type-aware training load proxy (mirrors AnalyticsEngine)."""
        atype = row.get("activity_type", "") if hasattr(row, "get") else getattr(row, "activity_type", "")
        dist = row.get("distance_km", 0) if hasattr(row, "get") else getattr(row, "distance_km", 0)
        elev = row.get("elevation_gain", 0) if hasattr(row, "get") else getattr(row, "elevation_gain", 0)
        mt = row.get("moving_time", 0) if hasattr(row, "get") else getattr(row, "moving_time", 0)
        min_ = (mt or 0) / 60.0

        if atype == "Run":
            return (dist or 0) * (1 + (elev or 0) / 1000.0)
        if atype == "Ride":
            return (dist or 0) * 0.5 * (1 + (elev or 0) / 2000.0)
        if atype == "Walk":
            return (dist or 0) * 0.3
        if atype == "Hike":
            return (dist or 0) * 0.6 * (1 + (elev or 0) / 1000.0)
        factor = self._DURATION_LOAD_PER_MIN.get(atype, 0.10)
        return min_ * factor

    def _daily_load_series(self, df: pd.DataFrame, days: int) -> pd.Series:
        """Daily load reindexed over the last `days` days (zeros for rest days)."""
        now = pd.Timestamp(datetime.utcnow())
        full_index = pd.date_range(end=now.normalize(), periods=days, freq="D")
        if df.empty:
            return pd.Series(0.0, index=full_index)
        sub = df[df["start_date"] >= now - timedelta(days=days)].copy()
        sub["_load"] = sub.apply(self._activity_load, axis=1)
        daily = sub.set_index("start_date").resample("D")["_load"].sum()
        return daily.reindex(full_index, fill_value=0.0)

    # ------------------------------------------------------------------
    # Banister-inspired load metrics
    # ------------------------------------------------------------------

    def monotony_score(self) -> float:
        """
        Training monotony = mean_daily_load / std_daily_load over 28 days.

        Interpretation:
          < 1.5 : well-periodised
          1.5-2.0: moderate monotony
          > 2.0 : high monotony — injury and overtraining risk
        """
        daily = self._daily_load_series(self.df_all, 28)
        mean = float(daily.mean())
        std = float(daily.std())
        if std < 0.01:
            return 0.0
        return round(mean / std, 2)

    def strain_score(self) -> float:
        """
        Banister strain = 7-day load × monotony.
        High strain indicates accumulated fatigue with insufficient variation.
        """
        daily_7 = self._daily_load_series(self.df_all, 7)
        weekly_load = float(daily_7.sum())
        return round(weekly_load * self.monotony_score(), 1)

    def consistency_score(self) -> float:
        """
        Consistency = % of weeks in the last 8 weeks containing at least 1 run.
        Range 0–100. Well-trained athletes typically score > 85.
        """
        if self.df_runs.empty:
            return 0.0
        now = pd.Timestamp(datetime.utcnow())
        cutoff = now - timedelta(weeks=8)
        recent = self.df_runs[self.df_runs["start_date"] >= cutoff]
        if recent.empty:
            return 0.0
        active_weeks = recent["week"].nunique()
        return round((active_weeks / 8.0) * 100.0, 1)

    def training_density(self) -> float:
        """Average number of activities per week over the last 4 weeks."""
        if self.df_all.empty:
            return 0.0
        now = pd.Timestamp(datetime.utcnow())
        recent = self.df_all[self.df_all["start_date"] >= now - timedelta(weeks=4)]
        return round(len(recent) / 4.0, 1)

    def recovery_debt(self) -> float:
        """
        Recovery debt = (ATL - CTL) / CTL.

        Positive: trained more than chronic average (accumulated fatigue).
        Negative: below chronic average (fresh/detrained).
        Typical range: -0.5 to +0.8.
        """
        daily = self._daily_load_series(self.df_all, 28)
        atl = float(daily[-7:].sum())
        ctl = float(daily.sum()) / 4.0
        if ctl < 0.01:
            return 0.0
        return round((atl - ctl) / ctl, 2)

    # ------------------------------------------------------------------
    # Aerobic fitness & efficiency
    # ------------------------------------------------------------------

    def aerobic_fitness_score(self) -> float:
        """
        Proxy for aerobic fitness based on pace-to-HR efficiency over the last
        8 weeks of runs.  Higher = more fit (faster pace at lower HR).

        Normalised to 0–100 using plausible athlete efficiency range.
        Returns 0 when fewer than 3 eligible runs.
        """
        if self.df_runs.empty:
            return 0.0
        now = pd.Timestamp(datetime.utcnow())
        df = self.df_runs[
            (self.df_runs["start_date"] >= now - timedelta(weeks=8))
            & self.df_runs["average_heartrate"].notna()
            & (self.df_runs["average_heartrate"] > 80)
            & self.df_runs["pace_min_per_km"].notna()
            & (self.df_runs["pace_min_per_km"] > 0)
            & ~np.isnan(self.df_runs["pace_min_per_km"])
            & (self.df_runs["distance_km"] >= 3)
        ].copy()
        if len(df) < 3:
            return 0.0
        # Efficiency: speed (km/min) per unit HR
        # = (1/pace) / (hr/100) → higher is better
        df["efficiency"] = (1.0 / df["pace_min_per_km"]) / (df["average_heartrate"] / 100.0)
        eff = float(df["efficiency"].mean())
        # Calibration: ~0.20 = beginner, ~0.40 = good club runner, ~0.55 = elite
        score = min(100.0, max(0.0, (eff - 0.12) / (0.48 - 0.12) * 100.0))
        return round(score, 1)

    def hr_efficiency_trend(self) -> Dict[str, Any]:
        """
        HR per km trend over the last 20 eligible runs.
        Declining = improving aerobic fitness (lower HR for same pace/distance).

        Returns:
          trend: 'improving' | 'declining' | 'stable'
          slope: regression slope (bpm/km per run)
          data: list[dict] for charting
        """
        empty: Dict[str, Any] = {"trend": "stable", "slope": 0.0, "data": []}
        if self.df_runs.empty:
            return empty
        df = self.df_runs[
            self.df_runs["average_heartrate"].notna()
            & (self.df_runs["distance_km"] >= 3)
        ].head(20).copy().sort_values("start_date")
        if len(df) < 4:
            return empty
        df["hr_per_km"] = df["average_heartrate"] / df["distance_km"]
        x = np.arange(len(df), dtype=float)
        slope = float(np.polyfit(x, df["hr_per_km"].values, 1)[0])
        if slope < -0.3:
            trend = "improving"
        elif slope > 0.3:
            trend = "declining"
        else:
            trend = "stable"
        chart_data = df[["start_date", "hr_per_km", "name"]].copy()
        chart_data["start_date"] = chart_data["start_date"].dt.strftime("%Y-%m-%d")
        return {
            "trend": trend,
            "slope": round(slope, 3),
            "data": chart_data.to_dict("records"),
        }

    def pace_efficiency_trend(self) -> Dict[str, Any]:
        """
        Pace-per-unit-HR trend over the last 20 eligible runs.
        Declining = more pace for the same HR (improving fitness).

        Returns:
          trend: 'improving' | 'declining' | 'stable'
          slope: regression slope
          data: list[dict] for charting
        """
        empty: Dict[str, Any] = {"trend": "stable", "slope": 0.0, "data": []}
        if self.df_runs.empty:
            return empty
        df = self.df_runs[
            self.df_runs["average_heartrate"].notna()
            & (self.df_runs["average_heartrate"] > 80)
            & self.df_runs["pace_min_per_km"].notna()
            & ~np.isnan(self.df_runs["pace_min_per_km"])
            & (self.df_runs["distance_km"] >= 3)
        ].head(20).copy().sort_values("start_date")
        if len(df) < 4:
            return empty
        df["pace_per_hr_unit"] = df["pace_min_per_km"] / (df["average_heartrate"] / 100.0)
        x = np.arange(len(df), dtype=float)
        slope = float(np.polyfit(x, df["pace_per_hr_unit"].values, 1)[0])
        trend = "improving" if slope < -0.02 else ("declining" if slope > 0.02 else "stable")
        chart_data = df[["start_date", "pace_per_hr_unit", "average_heartrate", "pace_min_per_km", "name"]].copy()
        chart_data["start_date"] = chart_data["start_date"].dt.strftime("%Y-%m-%d")
        return {
            "trend": trend,
            "slope": round(slope, 4),
            "data": chart_data.to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Training state detectors
    # ------------------------------------------------------------------

    def detect_overreaching(self) -> bool:
        """
        Overreaching: ACWR > 1.3 AND high strain score.
        Short-term: recoverable in days–weeks. Not the same as overtraining.
        """
        daily = self._daily_load_series(self.df_all, 28)
        atl = float(daily[-7:].sum())
        ctl = float(daily.sum()) / 4.0
        if ctl < 0.01:
            return False
        acwr = atl / ctl
        return acwr > 1.3 and self.strain_score() > 40

    def detect_undertraining(self) -> bool:
        """
        Undertraining: ACWR < 0.8 for 2 consecutive rolling 7-day windows.
        Suggests training volume is insufficient to maintain fitness.
        """
        now = pd.Timestamp(datetime.utcnow())
        for offset in [0, 1]:
            start = now - timedelta(days=7 * (offset + 1))
            end = now - timedelta(days=7 * offset)
            wdf = self.df_all[(self.df_all["start_date"] >= start) & (self.df_all["start_date"] < end)]
            w_load = float(wdf.apply(self._activity_load, axis=1).sum()) if not wdf.empty else 0.0
            daily28 = self._daily_load_series(self.df_all, 28)
            ctl = float(daily28.sum()) / 4.0
            if ctl > 0 and (w_load / ctl) >= 0.8:
                return False
        return True

    def detect_detraining(self) -> bool:
        """
        Detraining: load dropped > 25% over the past 3 weeks vs the prior 3 weeks.
        Fitness begins to decline noticeably after 2–3 weeks of reduced training.
        """
        if self.df_all.empty:
            return False
        now = pd.Timestamp(datetime.utcnow())
        recent = self.df_all[self.df_all["start_date"] >= now - timedelta(weeks=3)]
        prior = self.df_all[
            (self.df_all["start_date"] >= now - timedelta(weeks=6))
            & (self.df_all["start_date"] < now - timedelta(weeks=3))
        ]
        load_r = float(recent.apply(self._activity_load, axis=1).sum()) if not recent.empty else 0.0
        load_p = float(prior.apply(self._activity_load, axis=1).sum()) if not prior.empty else 0.0
        if load_p < 0.01:
            return False
        return (load_p - load_r) / load_p > 0.25

    # ------------------------------------------------------------------
    # Long run analysis
    # ------------------------------------------------------------------

    def long_run_pace_fade(self) -> Dict[str, Any]:
        """
        Analyse pace fade in long runs (> 12 km) using km splits.

        Compares average split time in the first third vs last third of
        the run.  Positive fade_pct = slowing down (common endurance limiter).

        Returns:
          fade_pct:  % pace deterioration (positive = slowing)
          verdict:   'negative_split' | 'even_pacing' | 'moderate_fade' | 'significant_fade'
          run_name:  name of most recent long run analysed
          runs_analysed: count
        """
        empty: Dict[str, Any] = {
            "fade_pct": None,
            "verdict": "insufficient_data",
            "run_name": None,
            "runs_analysed": 0,
        }
        if self.df_runs.empty:
            return empty

        long_runs = self.df_runs[
            (self.df_runs["distance_km"] > 12)
            & self.df_runs["splits_metric"].notna()
        ].head(5)
        if long_runs.empty:
            return empty

        fades: List[float] = []
        for _, row in long_runs.iterrows():
            try:
                splits = json.loads(row["splits_metric"])
                if not isinstance(splits, list) or len(splits) < 4:
                    continue
                # Only use splits with sufficient distance (> 500 m)
                times = [
                    s["moving_time"]
                    for s in splits
                    if s.get("moving_time") and (s.get("distance") or 0) > 500
                ]
                if len(times) < 4:
                    continue
                n3 = max(1, len(times) // 3)
                first_avg = sum(times[:n3]) / n3
                last_avg = sum(times[-n3:]) / n3
                if first_avg > 0:
                    fades.append((last_avg - first_avg) / first_avg * 100.0)
            except Exception:
                continue

        if not fades:
            return empty

        avg_fade = float(np.mean(fades))
        if avg_fade < 0:
            verdict = "negative_split"
        elif avg_fade <= 5:
            verdict = "even_pacing"
        elif avg_fade <= 10:
            verdict = "moderate_fade"
        else:
            verdict = "significant_fade"

        return {
            "fade_pct": round(avg_fade, 1),
            "verdict": verdict,
            "run_name": str(long_runs.iloc[0]["name"]),
            "runs_analysed": len(fades),
        }

    def long_run_hr_decoupling(self) -> Dict[str, Any]:
        """
        HR decoupling in long runs: compare average HR in first half vs second half.
        High decoupling (> 5%) suggests cardiovascular drift / poor endurance.
        """
        empty: Dict[str, Any] = {"decoupling_pct": None, "verdict": "insufficient_data"}
        if self.df_runs.empty:
            return empty

        long_runs = self.df_runs[
            (self.df_runs["distance_km"] > 12)
            & self.df_runs["splits_metric"].notna()
        ].head(5)
        if long_runs.empty:
            return empty

        decouplings: List[float] = []
        for _, row in long_runs.iterrows():
            try:
                splits = json.loads(row["splits_metric"])
                if not isinstance(splits, list) or len(splits) < 4:
                    continue
                hrs = [
                    s["average_heartrate"]
                    for s in splits
                    if s.get("average_heartrate") and s.get("average_heartrate") > 60
                ]
                if len(hrs) < 4:
                    continue
                mid = len(hrs) // 2
                first_hr = sum(hrs[:mid]) / mid
                last_hr = sum(hrs[mid:]) / (len(hrs) - mid)
                if first_hr > 0:
                    decouplings.append((last_hr - first_hr) / first_hr * 100.0)
            except Exception:
                continue

        if not decouplings:
            return empty

        avg_dec = float(np.mean(decouplings))
        if avg_dec < 3:
            verdict = "excellent"
        elif avg_dec < 5:
            verdict = "good"
        elif avg_dec < 8:
            verdict = "moderate_drift"
        else:
            verdict = "high_drift"

        return {"decoupling_pct": round(avg_dec, 1), "verdict": verdict}

    # ------------------------------------------------------------------
    # Cadence analysis
    # ------------------------------------------------------------------

    def cadence_trend(self) -> Dict[str, Any]:
        """
        Cadence trend over last 20 runs with cadence data.
        Target: 170–180 spm.  Improving = increasing toward target.
        """
        empty: Dict[str, Any] = {"trend": "stable", "avg_cadence": None, "data": []}
        if self.df_runs.empty:
            return empty
        df = self.df_runs[
            self.df_runs["average_cadence"].notna()
            & (self.df_runs["average_cadence"] > 50)
        ].head(20).copy().sort_values("start_date")
        if len(df) < 3:
            return empty
        avg_cad = float(df["average_cadence"].mean())
        x = np.arange(len(df), dtype=float)
        slope = float(np.polyfit(x, df["average_cadence"].values, 1)[0])
        # Improving = moving toward 170-180 range
        if avg_cad < 170 and slope > 0.1:
            trend = "improving"
        elif avg_cad > 180 and slope < -0.1:
            trend = "improving"
        elif slope < -0.2:
            trend = "declining"
        else:
            trend = "stable"
        chart_data = df[["start_date", "average_cadence", "name"]].copy()
        chart_data["start_date"] = chart_data["start_date"].dt.strftime("%Y-%m-%d")
        return {
            "trend": trend,
            "avg_cadence": round(avg_cad, 1),
            "slope": round(slope, 3),
            "data": chart_data.to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Monthly consistency heatmap data
    # ------------------------------------------------------------------

    def weekly_consistency_data(self, weeks: int = 16) -> List[Dict[str, Any]]:
        """
        Return per-week stats for a consistency heatmap.
        Used by the Training Intelligence tab.
        """
        if self.df_all.empty:
            return []
        now = pd.Timestamp(datetime.utcnow())
        cutoff = now - timedelta(weeks=weeks)
        df = self.df_all[self.df_all["start_date"] >= cutoff].copy()
        if df.empty:
            return []
        df["_load"] = df.apply(self._activity_load, axis=1)
        agg = (
            df.groupby("week")
            .agg(
                activity_count=("strava_id", "count"),
                total_load=("_load", "sum"),
                run_count=("activity_type", lambda x: (x == "Run").sum()),
            )
            .reset_index()
            .sort_values("week")
        )
        return agg.to_dict("records")

    # ------------------------------------------------------------------
    # Master compute
    # ------------------------------------------------------------------

    def compute_all(self) -> Dict[str, Any]:
        """Return all training state metrics as a single serialisable dict."""
        return {
            "monotony_score": self.monotony_score(),
            "strain_score": self.strain_score(),
            "consistency_score": self.consistency_score(),
            "training_density": self.training_density(),
            "recovery_debt": self.recovery_debt(),
            "aerobic_fitness_score": self.aerobic_fitness_score(),
            "hr_efficiency_trend": self.hr_efficiency_trend(),
            "pace_efficiency_trend": self.pace_efficiency_trend(),
            "cadence_analysis": self.cadence_trend(),
            "overreaching": self.detect_overreaching(),
            "undertraining": self.detect_undertraining(),
            "detraining": self.detect_detraining(),
            "long_run_pace_fade": self.long_run_pace_fade(),
            "long_run_hr_decoupling": self.long_run_hr_decoupling(),
            "weekly_consistency": self.weekly_consistency_data(),
        }
