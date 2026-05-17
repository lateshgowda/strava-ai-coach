import os
import time
from typing import Optional
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.db.crud import get_token, save_token

load_dotenv()

STRAVA_CLIENT_ID: str = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET: str = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_REDIRECT_URI: str = os.getenv(
    "STRAVA_REDIRECT_URI", "http://localhost:8000/auth/strava/callback"
)

STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"


def get_auth_url() -> str:
    """Build the Strava OAuth authorization URL."""
    params = {
        "client_id": STRAVA_CLIENT_ID,
        "redirect_uri": STRAVA_REDIRECT_URI,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": "activity:read_all",
    }
    return f"{STRAVA_AUTH_URL}?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Exchange an authorization code for an access + refresh token pair."""
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(STRAVA_TOKEN_URL, data=payload)
        response.raise_for_status()
        return response.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Obtain a new access token using the stored refresh token."""
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(STRAVA_TOKEN_URL, data=payload)
        response.raise_for_status()
        return response.json()


def is_token_expired(expires_at: int) -> bool:
    """Return True when the token has expired (with a 60-second safety margin)."""
    return time.time() > (expires_at - 60)


def get_valid_token(db: Session) -> Optional[str]:
    """
    Retrieve the stored access token, refreshing it if necessary.

    Returns the raw access_token string, or None if no token is stored.
    """
    token_record = get_token(db)
    if token_record is None:
        return None

    if is_token_expired(token_record.expires_at):
        try:
            new_token_data = refresh_access_token(token_record.refresh_token)
            # Preserve athlete metadata across refresh
            new_token_data["athlete_id"] = token_record.athlete_id
            new_token_data["athlete_name"] = token_record.athlete_name
            token_record = save_token(db, new_token_data)
        except httpx.HTTPError:
            # If refresh fails, the stored token may still work for a while
            pass

    return token_record.access_token
