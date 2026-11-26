"""Comprehensive unit, integration, and UI tests for FaceMask 360."""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from facemask_app import config, data, recognition


def test_verify_credentials_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config.verify_credentials("teacher1", "password123") is True
    assert config.verify_credentials("teacher1", "wrong") is False
    assert config.verify_credentials("unknown", "password123") is False

    monkeypatch.setenv("FACEMASK_USERS", "dean:topsecret,prof:alpha99")
    assert config.verify_credentials("dean", "topsecret") is True
    assert config.verify_credentials("teacher1", "password123") is False


def test_students_normalization_and_sorting(tmp_path: Path) -> None:
    csv_path = tmp_path / "students.csv"
    csv_path.write_text("Roll,Name,Class\n10,Rohan,F  \n2,Sandeep,Class A \n1,Ansh,A\n", encoding="utf-8")

    df = data.load_students(str(csv_path))
    assert df["Roll"].tolist() == ["1", "2", "10"]
    assert df["Class"].tolist() == ["A", "A", "F"]
    assert df["Name"].tolist() == ["Ansh", "Sandeep", "Rohan"]


def test_timetable_and_slot_detection() -> None:
    tt = data.load_timetable("timetable.csv")
    assert not tt.empty
    assert set(tt["Class"].unique()) == {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

    # Monday 09:30 -> Class A Slot 1 (09:00-10:00)
    monday_morning = datetime(2026, 11, 23, 9, 30, tzinfo=config.TIMEZONE)
    slot = data.get_current_slot(tt, "A", now=monday_morning)
    assert slot is not None
    assert slot["Slot"] == "Slot 1"

    # Sunday 23:00 -> no active slot
    sunday_night = datetime(2026, 11, 22, 23, 0, tzinfo=config.TIMEZONE)
    assert data.get_current_slot(tt, "A", now=sunday_night) is None
    assert data.get_current_slot(tt, "All", now=monday_morning) is None


def test_attendance_marking_idempotency_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "ATTENDANCE_DIR", str(tmp_path / "attendance"))
    students_df = pd.DataFrame(
        [
            {"Roll": "1", "Name": "Ansh", "Class": "A"},
            {"Roll": "2", "Name": "Sandeep", "Class": "A"},
            {"Roll": "3", "Name": "Aniruddha", "Class": "B"},
        ]
    )

    dets = [
        {"matched": True, "roll": "1", "name": "Ansh", "student_class": "A", "confidence": 0.91},
        {"matched": True, "roll": "3", "name": "Aniruddha", "student_class": "B", "confidence": 0.89},
        {"matched": False, "roll": "", "name": "Unknown", "student_class": "", "confidence": 0.40},
    ]

    newly, already, skipped = data.mark_attendance_records(
        dets, slot_name="Slot 1", class_filter="A", date_str="2026-11-26"
    )
    assert len(newly) == 1
    assert newly[0]["Roll"] == "1"
    assert len(already) == 0
    assert len(skipped) == 1
    assert skipped[0]["Roll"] == "3"

    # Re-marking the same student in the same slot must be idempotent
    newly2, already2, _ = data.mark_attendance_records(
        dets, slot_name="Slot 1", class_filter="A", date_str="2026-11-26"
    )
    assert len(newly2) == 0
    assert len(already2) == 1

    att_df = data.load_attendance("2026-11-26")
    assert len(att_df) == 1

    summary = data.compute_slot_summary(students_df, att_df, class_filter="A", slot_filter="Slot 1")
    assert summary["total"] == 2
    assert summary["present_count"] == 1
    assert summary["absent_count"] == 1
    assert summary["absent_df"]["Roll"].tolist() == ["2"]

    csv_bytes = data.to_csv_bytes(summary["absent_df"])
    xlsx_bytes = data.to_excel_bytes(summary["absent_df"], "Absent-A")
    assert b"Sandeep" in csv_bytes
    assert len(xlsx_bytes) > 100


def test_real_face_recognition_and_blank_rejection() -> None:
    students_df = data.load_students("students.csv")
    sample_path = Path("dataset/1_Ansh_A/1.jpg")
    assert sample_path.exists()

    img = Image.open(sample_path)
    annotated, detections = recognition.recognize_faces(img, students_df=students_df, threshold=0.72)
    assert annotated.size == img.size
    assert len(detections) >= 1
    assert detections[0]["matched"] is True
    assert detections[0]["roll"] == "1"
    assert detections[0]["name"] == "Ansh"
    assert detections[0]["student_class"] == "A"

    blank = Image.new("RGB", (200, 200), color=(240, 240, 240))
    _, blank_dets = recognition.recognize_faces(blank, students_df=students_df, threshold=0.72)
    assert blank_dets == []


def test_enroll_student_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recognition, "DATASET_DIR", str(tmp_path / "dataset"))
    emb_path = tmp_path / "embeddings.pkl"
    stu_path = tmp_path / "students.csv"
    with open(emb_path, "wb") as f:
        pickle.dump({"X": [], "y": [], "student_embeddings": {}}, f)

    sample_img = Image.open("dataset/1_Ansh_A/1.jpg")
    ok, msg = recognition.enroll_student_from_image(
        sample_img,
        roll="999",
        name="TestUser",
        student_class="A",
        embeddings_path=str(emb_path),
        students_path=str(stu_path),
    )
    assert ok is True
    assert "TestUser" in msg

    df = data.load_students(str(stu_path))
    assert df["Roll"].tolist() == ["999"]

    recognition.load_embeddings_index.clear()
    _, dets = recognition.recognize_faces(
        sample_img, students_df=df, threshold=0.72, embeddings_path=str(emb_path)
    )
    assert len(dets) == 1
    assert dets[0]["matched"] is True
    assert dets[0]["roll"] == "999"
    assert dets[0]["name"] == "TestUser"
    recognition.load_embeddings_index.clear()


@pytest.mark.parametrize("script", ["app.py", "app2.py", "dashboard.py"])
def test_streamlit_entrypoints_boot_cleanly(script: str) -> None:
    at = AppTest.from_file(script, default_timeout=30).run()
    assert len(at.exception) == 0
    assert len(at.error) == 0


def test_streamlit_navigation_and_auth_flow() -> None:
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert len(at.exception) == 0

    # Navigate to Timetable
    at.sidebar.radio[0].set_value("Timetable").run()
    assert len(at.exception) == 0

    # Navigate to Dashboard -> shows auth gate
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert len(at.exception) == 0
    assert len(at.text_input) >= 2

    # Sign in with wrong password -> error shown
    at.text_input[0].set_value("teacher1")
    at.text_input[1].set_value("wrongpass")
    sign_in_btn = [b for b in at.button if b.label == "Sign In"][0]
    sign_in_btn.click().run()
    assert len(at.error) == 1

    # Sign in with valid password -> unlocks dashboard tabs
    at.text_input[0].set_value("teacher1")
    at.text_input[1].set_value("password123")
    sign_in_btn = [b for b in at.button if b.label == "Sign In"][0]
    sign_in_btn.click().run()
    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.tabs) >= 1

    # Select historical date 2025-09-06 on Dashboard if present
    date_select = at.selectbox[0]
    if "2025-09-06" in date_select.options:
        date_select.set_value("2025-09-06").run()
        assert len(at.exception) == 0

    # Navigate to Students while authenticated
    at.sidebar.radio[0].set_value("Students").run()
    assert len(at.exception) == 0
    assert len(at.dataframe) >= 1

    # Sign out cleanly
    sign_out_btn = [b for b in at.sidebar.button if b.label == "Sign Out"][0]
    sign_out_btn.click().run()
    assert at.session_state["authenticated"] is False
