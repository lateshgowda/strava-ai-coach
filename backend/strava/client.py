from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "https://www.strava.com/api/v3"

_DEFAULT_TIMEOUT = 30.0


class StravaClient:
    """Thin wrapper around the Strava v3 REST API."""

    def __init__(self, access_token: str) -> None:
        self._headers = {"Authorization": f"Bearer {access_token}"}

    # ------------------------------------------------------------------
    # Athlete
    # ------------------------------------------------------------------

    def get_athlete(self) -> Dict[str, Any]:
        """Return the authenticated athlete's profile."""
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/athlete",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    def get_activities(
        self,
        before: Optional[int] = None,
        after: Optional[int] = None,
        per_page: int = 200,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Return a page of the athlete's activities.

        Parameters
        ----------
        before:   Unix timestamp — return only activities before this time.
        after:    Unix timestamp — return only activities after this time.
        per_page: Number of results per page (max 200).
        page:     Page number (1-based).
        """
        params: Dict[str, Any] = {
            "per_page": min(per_page, 200),
            "page": page,
        }
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/athlete/activities",
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            # Strava returns an error object on failure
            return []

    def get_activity_detail(self, activity_id: int) -> Dict[str, Any]:
        """
        Return the full detail for a single activity, including splits_metric.
        """
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/activities/{activity_id}",
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Paginated helper
    # ------------------------------------------------------------------

    def get_all_activities(
        self,
        after: Optional[int] = None,
        before: Optional[int] = None,
        per_page: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Iterate through all pages and return every activity.

        Stops when a page returns fewer items than *per_page*.
        """
        all_activities: List[Dict[str, Any]] = []
        page = 1
        while True:
            batch = self.get_activities(
                before=before,
                after=after,
                per_page=per_page,
                page=page,
            )
            if not batch:
                break
            all_activities.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return all_activities
