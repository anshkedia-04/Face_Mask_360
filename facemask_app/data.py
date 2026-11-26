"""Data loading, normalization, attendance persistence, and export helpers."""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd

from facemask_app.config import (
    ATTENDANCE_DIR,
    DATASET_DIR,
    DEFAULT_SLOTS,
    STUDENTS_FILE,
    TIMETABLE_FILE,
    now_ist,
)

ATTENDANCE_COLUMNS = ["Roll", "Name", "Class", "Slot", "Time", "Status"]


def ensure_directories() -> None:
    """Create required application directories if missing."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)


def normalize_class_code(value: Any) -> str:
    """Normalize class strings such as 'Class A' or 'A  ' to 'A'."""
    text = str(value).strip()
    if text.lower().startswith("class "):
        text = text[6:].strip()
    return text


def format_class_label(class_code: str) -> str:
    """Format a normalized class code for display."""
    clean = normalize_class_code(class_code)
    if not clean or clean.lower() == "all":
        return "All Classes"
    return f"Class {clean}"


def _roll_sort_key(roll: str) -> tuple[int, int | str]:
    text = str(roll).strip()
    return (0, int(text)) if text.isdigit() else (1, text)


def load_students(path: str = STUDENTS_FILE) -> pd.DataFrame:
    """Load and normalize the student roster CSV."""
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame(columns=["Roll", "Name", "Class"])

    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("Roll", "Name", "Class"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    df["Class"] = df["Class"].apply(normalize_class_code)
    df = df[df["Roll"] != ""].drop_duplicates(subset=["Roll"], keep="last")
    df = df.sort_values(by="Roll", key=lambda s: s.map(_roll_sort_key)).reset_index(drop=True)
    return df[["Roll", "Name", "Class"]]


def upsert_student_record(roll: str, name: str, student_class: str, path: str = STUDENTS_FILE) -> pd.DataFrame:
    """Insert or update a student row in students.csv and return the updated DataFrame."""
    df = load_students(path)
    clean_roll = str(roll).strip()
    clean_name = str(name).strip()
    clean_class = normalize_class_code(student_class)

    df = df[df["Roll"] != clean_roll]
    new_row = pd.DataFrame([{"Roll": clean_roll, "Name": clean_name, "Class": clean_class}])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values(by="Roll", key=lambda s: s.map(_roll_sort_key)).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def load_timetable(path: str = TIMETABLE_FILE) -> pd.DataFrame:
    """Load and normalize the weekly timetable CSV."""
    cols = ["Class", "Day", "Slot", "Start_Time", "End_Time"]
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path, dtype=str).fillna("")
    for col in cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    df["Class"] = df["Class"].apply(normalize_class_code)
    df["Day"] = df["Day"].str.capitalize()
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%H:%M", errors="coerce").dt.strftime("%H:%M")
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="%H:%M", errors="coerce").dt.strftime("%H:%M")
    return df[cols].dropna(subset=["Start_Time", "End_Time"]).reset_index(drop=True)


def get_current_slot(
    timetable_df: pd.DataFrame,
    class_code: str,
    now: datetime | None = None,
) -> dict[str, str] | None:
    """Return active slot metadata {'Slot', 'Start_Time', 'End_Time'} for a class, or None."""
    if timetable_df.empty:
        return None

    clean_class = normalize_class_code(class_code)
    if not clean_class or clean_class.lower() == "all":
        return None

    ref = now or now_ist()
    today_day = ref.strftime("%A").strip().capitalize()
    now_hhmm = ref.strftime("%H:%M")

    day_rows = timetable_df[
        (timetable_df["Class"] == clean_class) & (timetable_df["Day"] == today_day)
    ]
    for _, row in day_rows.iterrows():
        if row["Start_Time"] <= now_hhmm <= row["End_Time"]:
            return {
                "Slot": row["Slot"],
                "Start_Time": row["Start_Time"],
                "End_Time": row["End_Time"],
            }
    return None


def get_available_slots(timetable_df: pd.DataFrame, class_code: str | None = None) -> list[str]:
    """Return sorted slot names from timetable (with fallback to Slot 1..6)."""
    if not timetable_df.empty:
        subset = timetable_df
        if class_code and normalize_class_code(class_code).lower() != "all":
            filtered = timetable_df[timetable_df["Class"] == normalize_class_code(class_code)]
            if not filtered.empty:
                subset = filtered
        slots = sorted(subset["Slot"].dropna().unique().tolist())
        if slots:
            return slots
    return list(DEFAULT_SLOTS)


def get_attendance_path(date_str: str | None = None, create_dir: bool = False) -> str:
    """Return canonical attendance CSV path for YYYY-MM-DD."""
    day = date_str or now_ist().strftime("%Y-%m-%d")
    folder = os.path.join(ATTENDANCE_DIR, day)
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "attendance.csv")
    if not os.path.exists(path):
        legacy_path = os.path.join("Attendance", day, "attendance.csv")
        if os.path.exists(legacy_path):
            return legacy_path
    return path


def list_attendance_dates() -> list[str]:
    """List available YYYY-MM-DD attendance folders sorted newest first."""
    dates: set[str] = set()
    for base in (ATTENDANCE_DIR, "Attendance"):
        if os.path.isdir(base):
            for entry in os.listdir(base):
                full = os.path.join(base, entry, "attendance.csv")
                if os.path.isfile(full):
                    dates.add(entry)
    today = now_ist().strftime("%Y-%m-%d")
    dates.add(today)
    return sorted(dates, reverse=True)


def load_attendance(date_str: str | None = None) -> pd.DataFrame:
    """Load attendance CSV for a date with normalized string columns."""
    path = get_attendance_path(date_str, create_dir=False)
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ATTENDANCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df["Class"] = df["Class"].apply(normalize_class_code)
    return df[ATTENDANCE_COLUMNS]


def save_attendance(df: pd.DataFrame, date_str: str | None = None) -> str:
    """Persist attendance DataFrame to disk and return file path."""
    path = get_attendance_path(date_str, create_dir=True)
    df[ATTENDANCE_COLUMNS].to_csv(path, index=False)
    return path


def mark_attendance_records(
    detections: list[dict[str, Any]],
    slot_name: str,
    class_filter: str = "All",
    date_str: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Mark recognized students present idempotently.

    Returns (newly_marked, already_marked, skipped_other_class).
    """
    df = load_attendance(date_str)
    clean_filter = normalize_class_code(class_filter)
    filter_active = bool(clean_filter and clean_filter.lower() != "all")

    newly_marked: list[dict[str, str]] = []
    already_marked: list[dict[str, str]] = []
    skipped_other_class: list[dict[str, str]] = []

    timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
    new_rows: list[dict[str, str]] = []

    for det in detections:
        if not det.get("matched"):
            continue
        roll = str(det["roll"]).strip()
        name = str(det["name"]).strip()
        student_class = normalize_class_code(det["student_class"])

        if filter_active and student_class != clean_filter:
            skipped_other_class.append(
                {"Roll": roll, "Name": name, "Class": student_class}
            )
            continue

        exists_mask = (df["Roll"] == roll) & (df["Slot"] == slot_name)
        pending_exists = any(r["Roll"] == roll and r["Slot"] == slot_name for r in new_rows)
        if exists_mask.any() or pending_exists:
            already_marked.append({"Roll": roll, "Name": name, "Class": student_class})
            continue

        record = {
            "Roll": roll,
            "Name": name,
            "Class": student_class,
            "Slot": slot_name,
            "Time": timestamp,
            "Status": "Present",
        }
        new_rows.append(record)
        newly_marked.append(record)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_attendance(df, date_str)

    return newly_marked, already_marked, skipped_other_class


def compute_slot_summary(
    students_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    class_filter: str = "All",
    slot_filter: str = "All",
) -> dict[str, Any]:
    """Compute present/absent lists and KPI counts for a class and slot filter."""
    clean_class = normalize_class_code(class_filter)
    roster = students_df.copy()
    if clean_class and clean_class.lower() != "all":
        roster = roster[roster["Class"] == clean_class]

    att = attendance_df.copy()
    if clean_class and clean_class.lower() != "all":
        att = att[att["Class"] == clean_class]
    if slot_filter and slot_filter != "All":
        att = att[att["Slot"] == slot_filter]

    present_att = att[att["Status"].str.lower() == "present"].copy()
    present_rolls = set(present_att["Roll"].astype(str).str.strip())

    absent_df = roster[~roster["Roll"].isin(present_rolls)].copy().reset_index(drop=True)
    present_roster = roster[roster["Roll"].isin(present_rolls)].copy().reset_index(drop=True)

    total = len(roster)
    present_count = len(present_roster)
    absent_count = len(absent_df)
    rate = (present_count / total * 100.0) if total > 0 else 0.0

    return {
        "total": total,
        "present_count": present_count,
        "absent_count": absent_count,
        "rate": rate,
        "present_df": present_att.reset_index(drop=True),
        "absent_df": absent_df,
        "roster_df": roster.reset_index(drop=True),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Attendance") -> bytes:
    """Serialize a DataFrame as Excel (.xlsx) bytes."""
    output = BytesIO()
    safe_sheet = sheet_name[:31].replace("/", "-").replace("\\", "-")
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return output.getvalue()
