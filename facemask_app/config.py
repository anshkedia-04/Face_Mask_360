"""Configuration and authentication helpers for FaceMask 360."""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

DATASET_DIR = "dataset"
STUDENTS_FILE = "students.csv"
TIMETABLE_FILE = "timetable.csv"
ATTENDANCE_DIR = "attendance"
EMBEDDINGS_FILE = "student_embeddings.pkl"
LOGO_PATH = "Logo.png"

TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_SLOTS = [f"Slot {i}" for i in range(1, 7)]
DEFAULT_THRESHOLD = 0.72

DEFAULT_USERS = {
    "teacher1": "password123",
    "admin": "securepass",
}


def now_ist() -> datetime:
    """Return the current timezone-aware datetime in Asia/Kolkata."""
    return datetime.now(TIMEZONE)


def get_auth_users() -> dict[str, str]:
    """Load credentials from environment or Streamlit secrets, with demo fallback."""
    users = dict(DEFAULT_USERS)
    env_users = os.environ.get("FACEMASK_USERS", "").strip()
    if env_users:
        parsed: dict[str, str] = {}
        for pair in env_users.split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                if u.strip() and p.strip():
                    parsed[u.strip()] = p.strip()
        if parsed:
            return parsed

    try:
        import streamlit as st

        if "auth_users" in st.secrets:
            return {str(k): str(v) for k, v in st.secrets["auth_users"].items()}
    except Exception:
        pass

    return users


def verify_credentials(username: str, password: str) -> bool:
    """Validate username and password against configured users."""
    users = get_auth_users()
    clean_user = username.strip()
    return clean_user in users and users[clean_user] == password
