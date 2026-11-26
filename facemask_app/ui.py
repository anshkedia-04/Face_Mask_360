"""Smooth, simple, and elegant Streamlit UI for FaceMask 360."""

from __future__ import annotations

import os
from typing import Any

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from facemask_app.config import (
    DEFAULT_THRESHOLD,
    LOGO_PATH,
    now_ist,
    verify_credentials,
)
from facemask_app.data import (
    compute_slot_summary,
    ensure_directories,
    format_class_label,
    get_available_slots,
    get_current_slot,
    list_attendance_dates,
    load_attendance,
    load_students,
    load_timetable,
    mark_attendance_records,
    normalize_class_code,
    to_csv_bytes,
    to_excel_bytes,
)
from facemask_app.recognition import (
    enroll_student_from_image,
    load_embeddings_index,
    recognize_faces,
)

NAV_ITEMS = ["Mark Attendance", "Dashboard", "Timetable", "Students"]

CUSTOM_CSS = """
<style>
    /* Clean page spacing and typography */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer, [data-testid="stAppDeployButton"] {
        visibility: hidden;
    }

    /* Minimalist Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Brand & Cards */
    .fm-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.5rem;
    }
    .fm-brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .fm-brand-sub {
        font-size: 0.78rem;
        color: #64748B;
    }

    .fm-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    .fm-kpi {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        min-height: 114px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .fm-kpi-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
        word-break: keep-all;
        white-space: nowrap;
    }
    .fm-kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .fm-kpi-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Status Pills */
    .fm-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.3;
    }
    .fm-pill-green {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .fm-pill-amber {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .fm-pill-indigo {
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .fm-pill-slate {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    /* Header bar */
    .fm-page-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .fm-page-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }
    .fm-page-desc {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Refined Buttons & Inputs */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        white-space: nowrap;
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] > div {
        gap: 0.3rem;
    }
    .fm-checkin-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.65rem;
        border-radius: 10px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.45rem;
        font-size: 0.86rem;
    }
</style>
"""


def _init_session_state(default_page: str) -> None:
    defaults: dict[str, Any] = {
        "authenticated": False,
        "auth_user": None,
        "nav_page": default_page if default_page in NAV_ITEMS else "Mark Attendance",
        "webcam_running": False,
        "recent_checkins": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_page_header(title: str, description: str, badge_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="fm-page-header">
            <div>
                <h1 class="fm-page-title">{title}</h1>
                <div class="fm-page-desc">{description}</div>
            </div>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi(col: Any, label: str, value: str, subtext: str) -> None:
    col.markdown(
        f"""
        <div class="fm-kpi">
            <div class="fm-kpi-label">{label}</div>
            <div class="fm-kpi-value">{value}</div>
            <div class="fm-kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_auth_gate(section_label: str) -> bool:
    """Show a sleek inline sign-in card if not authenticated. Returns auth status."""
    if st.session_state.get("authenticated"):
        return True

    left, center, right = st.columns([1, 1.4, 1])
    with center:
        st.markdown(
            f"""
            <div class="fm-card">
                <div style="font-size:1.05rem;font-weight:700;color:#0F172A;margin-bottom:0.25rem;">
                    Teacher Sign-In Required
                </div>
                <div style="font-size:0.86rem;color:#64748B;">
                    Sign in to access {section_label}, export reports, and manage records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form(key=f"login_form_{section_label}", clear_on_submit=False):
            username = st.text_input("Username", placeholder="teacher1")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", type="primary", width="stretch")
            if submitted:
                if verify_credentials(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["auth_user"] = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption("Demo credentials: `teacher1` / `password123` (or `admin` / `securepass`).")
    return False


def _render_sidebar(students_df: pd.DataFrame) -> str:
    with st.sidebar:
        logo_col, text_col = st.columns([1, 3.2])
        with logo_col:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=44)
            else:
                st.markdown("<div style='font-size:1.8rem;'>🎯</div>", unsafe_allow_html=True)
        with text_col:
            st.markdown(
                """
                <div class="fm-brand-title">FaceMask 360</div>
                <div class="fm-brand-sub">FaceNet Attendance</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        selected_page = st.radio(
            "Navigation",
            NAV_ITEMS,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")
        now = now_ist()
        emb_index = load_embeddings_index()
        enrolled_count = len(emb_index["keys"])
        roster_count = len(students_df)

        st.markdown(
            f"""
            <div style="font-size:0.82rem;color:#475569;line-height:1.6;">
                <div><strong>{now.strftime("%a, %d %b %Y")}</strong></div>
                <div>{now.strftime("%I:%M %p")} IST</div>
                <div style="margin-top:0.4rem;">
                    <span class="fm-pill fm-pill-indigo">{enrolled_count}/{roster_count} Faces Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if st.session_state.get("authenticated"):
            st.caption(f"Signed in as **{st.session_state.get('auth_user')}**")
            if st.button("Sign Out", width="stretch"):
                st.session_state["authenticated"] = False
                st.session_state["auth_user"] = None
                st.rerun()
        else:
            st.caption("Public Kiosk Mode · Sign in on Dashboard or Students tab.")

    return selected_page


def _process_and_display_frame(
    pil_img: Image.Image,
    students_df: pd.DataFrame,
    threshold: float,
    selected_class: str,
    active_slot: str | None,
) -> None:
    """Run FaceNet recognition on a single image, mark attendance, and show feedback."""
    annotated, detections = recognize_faces(
        pil_img,
        students_df=students_df,
        threshold=threshold,
    )
    st.image(annotated, width="stretch")

    if not detections:
        st.info("No face detected. Ensure good lighting and face the camera directly.")
        return

    matched_dets = [d for d in detections if d["matched"]]
    unknown_count = len(detections) - len(matched_dets)

    if not matched_dets:
        st.warning(f"Detected {unknown_count} unrecognized face(s) below {threshold:.0%} confidence.")
        return

    if not active_slot:
        names = ", ".join(f"{d['name']} ({d['roll']})" for d in matched_dets)
        st.warning(f"Recognized **{names}**, but no slot is selected. Choose a Manual Slot to record attendance.")
        return

    newly, already, other_class = mark_attendance_records(
        matched_dets,
        slot_name=active_slot,
        class_filter=selected_class,
    )

    for rec in newly:
        entry = f"{rec['Name']} (Roll {rec['Roll']} · Class {rec['Class']})"
        st.success(f"Marked Present for {active_slot}: **{entry}**")
        st.session_state["recent_checkins"].insert(
            0,
            {
                "name": rec["Name"],
                "roll": rec["Roll"],
                "class": rec["Class"],
                "slot": rec["Slot"],
                "time": rec["Time"].split(" ")[-1],
                "status": "New",
            },
        )
        st.session_state["recent_checkins"] = st.session_state["recent_checkins"][:8]

    for rec in already:
        st.info(f"Already marked in {active_slot}: **{rec['Name']}** (Roll {rec['Roll']})")

    for rec in other_class:
        st.warning(
            f"Recognized **{rec['Name']}** (Class {rec['Class']}), which differs from selected {format_class_label(selected_class)}."
        )


def _render_recent_checkins(
    placeholder: Any,
    students_df: pd.DataFrame,
    selected_class: str,
    active_slot: str | None,
) -> None:
    """Render live check-in feed and slot count into the reserved placeholder."""
    with placeholder.container():
        st.markdown("#### Live Check-ins")
        today_df = load_attendance()
        if active_slot:
            summary = compute_slot_summary(students_df, today_df, selected_class, active_slot)
            st.caption(
                f"{format_class_label(selected_class)} · {active_slot}: "
                f"**{summary['present_count']}** present / **{summary['total']}** total"
            )

        checkins = st.session_state.get("recent_checkins", [])
        if checkins:
            for item in checkins[:5]:
                st.markdown(
                    f"""
                    <div class="fm-checkin-row">
                        <div><strong>{item['name']}</strong> <span style="color:#64748B;">#{item['roll']} · Class {item['class']}</span></div>
                        <div style="color:#065F46;font-weight:600;">{item['time']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No check-ins recorded in this session yet.")


def _render_mark_attendance(students_df: pd.DataFrame, timetable_df: pd.DataFrame) -> None:
    classes = sorted(students_df["Class"].dropna().unique().tolist())
    class_options = ["All"] + [c for c in classes if c]
    threshold = DEFAULT_THRESHOLD

    left_col, right_col = st.columns([1.35, 1.05], gap="medium")

    with right_col:
        st.markdown("#### Session Setup")
        selected_class = st.selectbox(
            "Class",
            class_options,
            format_func=format_class_label,
            key="att_class_select",
        )

        auto_slot = get_current_slot(timetable_df, selected_class)
        slot_mode = st.radio(
            "Slot Selection",
            ["Auto (Timetable)", "Manual Slot"],
            horizontal=True,
            key="att_slot_mode",
        )

        if slot_mode == "Auto (Timetable)":
            if auto_slot:
                active_slot = auto_slot["Slot"]
                st.markdown(
                    f"<span class='fm-pill fm-pill-green'>Active: {active_slot} ({auto_slot['Start_Time']}–{auto_slot['End_Time']})</span>",
                    unsafe_allow_html=True,
                )
            elif selected_class == "All":
                active_slot = None
                st.markdown(
                    "<span class='fm-pill fm-pill-amber'>Select a class or Manual Slot</span>",
                    unsafe_allow_html=True,
                )
            else:
                active_slot = None
                st.markdown(
                    "<span class='fm-pill fm-pill-amber'>No active slot · Switch to Manual</span>",
                    unsafe_allow_html=True,
                )
        else:
            available_slots = get_available_slots(timetable_df, selected_class)
            active_slot = st.selectbox("Select Slot", available_slots, key="att_manual_slot")

        with st.expander("Match Threshold", expanded=False):
            threshold = st.slider(
                "Minimum Confidence",
                min_value=0.55,
                max_value=0.90,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                help="Minimum cosine similarity score required to mark a student present.",
            )

        recent_placeholder = st.empty()

    with left_col:
        st.markdown("#### Camera & Capture")
        capture_mode = st.radio(
            "Input Mode",
            ["Browser Camera", "Upload Photo", "Local Webcam"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if capture_mode != "Local Webcam" and st.session_state.get("webcam_running"):
            st.session_state["webcam_running"] = False

        if capture_mode == "Browser Camera":
            cam_file = st.camera_input("Take a photo to mark attendance", label_visibility="collapsed")
            if cam_file is not None:
                img = Image.open(cam_file)
                _process_and_display_frame(img, students_df, threshold, selected_class, active_slot)

        elif capture_mode == "Upload Photo":
            uploaded = st.file_uploader(
                "Upload a student photo (JPG or PNG)",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                img = Image.open(uploaded)
                _process_and_display_frame(img, students_df, threshold, selected_class, active_slot)

        else:
            c1, c2 = st.columns(2)
            if c1.button("Start Webcam", type="primary", width="stretch", disabled=st.session_state["webcam_running"]):
                st.session_state["webcam_running"] = True
                st.rerun()
            if c2.button("Stop Webcam", width="stretch", disabled=not st.session_state["webcam_running"]):
                st.session_state["webcam_running"] = False
                st.rerun()

            if st.session_state["webcam_running"]:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.session_state["webcam_running"] = False
                    st.warning("Local camera device could not be opened on this host. Use **Browser Camera** or **Upload Photo** instead.")
                else:
                    frame_slot = st.empty()
                    status_slot = st.empty()
                    frame_idx = 0
                    try:
                        while st.session_state.get("webcam_running") and cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                status_slot.error("Failed to read frame from local webcam.")
                                break
                            frame_idx += 1
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_frame = Image.fromarray(rgb)
                            if frame_idx % 2 == 1:
                                annotated, detections = recognize_faces(
                                    pil_frame,
                                    students_df=students_df,
                                    threshold=threshold,
                                )
                                frame_slot.image(annotated, width="stretch")
                                matched = [d for d in detections if d["matched"]]
                                if matched and active_slot:
                                    newly, _, _ = mark_attendance_records(
                                        matched,
                                        slot_name=active_slot,
                                        class_filter=selected_class,
                                    )
                                    if newly:
                                        names = ", ".join(f"{r['Name']} ({r['Roll']})" for r in newly)
                                        status_slot.success(f"Marked Present: {names}")
                            else:
                                frame_slot.image(pil_frame, width="stretch")
                    finally:
                        cap.release()

    _render_recent_checkins(recent_placeholder, students_df, selected_class, active_slot)


def _render_dashboard(students_df: pd.DataFrame, timetable_df: pd.DataFrame) -> None:
    dates = list_attendance_dates()
    classes = sorted(students_df["Class"].dropna().unique().tolist())
    slots = get_available_slots(timetable_df)

    f1, f2, f3, f4 = st.columns([1.3, 1.35, 1.0, 0.9], gap="small")
    selected_date = f1.selectbox("Date", dates, index=0)
    selected_class = f2.selectbox("Class", ["All"] + classes, format_func=format_class_label, key="dash_class")
    selected_slot = f3.selectbox("Slot", ["All"] + slots, key="dash_slot")
    with f4:
        st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
        if st.button("Refresh", width="stretch"):
            st.rerun()

    att_df = load_attendance(selected_date)
    summary = compute_slot_summary(students_df, att_df, selected_class, selected_slot)

    k1, k2, k3, k4 = st.columns(4)
    _render_kpi(k1, "Present", str(summary["present_count"]), f"On {selected_date}")
    _render_kpi(k2, "Absent", str(summary["absent_count"]), "Not yet marked")
    _render_kpi(k3, "Roster Total", str(summary["total"]), format_class_label(selected_class))
    _render_kpi(k4, "Attendance %", f"{summary['rate']:.1f}%", selected_slot)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    if not _render_auth_gate("Attendance Reports"):
        return

    search = st.text_input("Filter by student name or roll number", placeholder="Type a name or roll...")

    present_df = summary["present_df"]
    absent_df = summary["absent_df"]

    if search.strip():
        q = search.strip().lower()
        present_df = present_df[
            present_df["Name"].str.lower().str.contains(q) | present_df["Roll"].str.lower().str.contains(q)
        ]
        absent_df = absent_df[
            absent_df["Name"].str.lower().str.contains(q) | absent_df["Roll"].str.lower().str.contains(q)
        ]

    tab_present, tab_absent, tab_all = st.tabs(
        [
            f"Present ({len(present_df)})",
            f"Absent ({len(absent_df)})",
            f"All Logs ({len(att_df)})",
        ]
    )

    with tab_present:
        if present_df.empty:
            st.info("No present records match the current filters.")
        else:
            st.dataframe(present_df, width="stretch", hide_index=True)
            c1, c2, _ = st.columns([1.15, 1.35, 1.5])
            c1.download_button(
                "Export CSV",
                data=to_csv_bytes(present_df),
                file_name=f"present_{selected_class}_{selected_date}.csv",
                mime="text/csv",
                width="stretch",
            )
            c2.download_button(
                "Export Excel",
                data=to_excel_bytes(present_df, f"Present-{selected_class}"),
                file_name=f"present_{selected_class}_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    with tab_absent:
        if absent_df.empty:
            st.success("Everyone in this selection is marked present.")
        else:
            st.dataframe(absent_df, width="stretch", hide_index=True)
            c1, c2, _ = st.columns([1.15, 1.35, 1.5])
            c1.download_button(
                "Export CSV",
                data=to_csv_bytes(absent_df),
                file_name=f"absent_{selected_class}_{selected_date}.csv",
                mime="text/csv",
                width="stretch",
            )
            c2.download_button(
                "Export Excel",
                data=to_excel_bytes(absent_df, f"Absent-{selected_class}"),
                file_name=f"absent_{selected_class}_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    with tab_all:
        if att_df.empty:
            st.info(f"No attendance log file exists yet for {selected_date}.")
        else:
            st.dataframe(att_df, width="stretch", hide_index=True)


def _render_timetable(timetable_df: pd.DataFrame) -> None:
    if timetable_df.empty:
        st.info("No timetable data found in `timetable.csv`.")
        return

    classes = sorted(timetable_df["Class"].dropna().unique().tolist())
    today_name = now_ist().strftime("%A")

    c1, c2 = st.columns(2)
    selected_class = c1.selectbox("Select Class", classes, format_func=format_class_label, key="tt_class")
    class_df = timetable_df[timetable_df["Class"] == selected_class]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    available_days = [d for d in day_order if d in set(class_df["Day"])]
    default_idx = available_days.index(today_name) + 1 if today_name in available_days else 0
    selected_day = c2.selectbox("Day", ["All Days"] + available_days, index=default_idx, key="tt_day")

    active = get_current_slot(timetable_df, selected_class)
    if active:
        st.markdown(
            f"<span class='fm-pill fm-pill-green'>Live Now: {active['Slot']} ({active['Start_Time']}–{active['End_Time']})</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    days_to_show = available_days if selected_day == "All Days" else [selected_day]
    for day in days_to_show:
        day_df = class_df[class_df["Day"] == day][["Slot", "Start_Time", "End_Time"]].sort_values("Start_Time")
        if day_df.empty:
            continue
        is_today = day == today_name
        label = f"{day} (Today)" if is_today else day
        st.markdown(f"#### {label}")
        st.dataframe(day_df.reset_index(drop=True), width="stretch", hide_index=True)


def _render_students(students_df: pd.DataFrame) -> None:
    if not _render_auth_gate("Student Management"):
        return

    tab_roster, tab_enroll = st.tabs(["Student Roster", "Enroll New Student"])

    with tab_roster:
        classes = sorted(students_df["Class"].dropna().unique().tolist())
        c1, c2 = st.columns([1, 2])
        selected_class = c1.selectbox("Filter Class", ["All"] + classes, format_func=format_class_label, key="stu_class")
        query = c2.text_input("Search", placeholder="Search by roll or name...", key="stu_search")

        emb_keys = set(load_embeddings_index()["keys"])
        enrolled_rolls = {k.split("_")[0] for k in emb_keys}

        view_df = students_df.copy()
        if selected_class != "All":
            view_df = view_df[view_df["Class"] == selected_class]
        if query.strip():
            q = query.strip().lower()
            view_df = view_df[view_df["Name"].str.lower().str.contains(q) | view_df["Roll"].str.lower().str.contains(q)]

        view_df = view_df.copy()
        view_df["Face Embedding"] = view_df["Roll"].map(lambda r: "Ready" if r in enrolled_rolls else "Missing")
        st.caption(f"Showing **{len(view_df)}** students")
        st.dataframe(view_df, width="stretch", hide_index=True)

    with tab_enroll:
        st.caption("Add a new student (or update an existing roll's photo) and immediately refresh FaceNet embeddings.")
        classes = sorted(set(students_df["Class"].dropna().tolist()) | {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"})

        f_left, f_right = st.columns([1, 1.15], gap="large")
        with f_left:
            roll = st.text_input("Roll Number", placeholder="e.g., 101")
            name = st.text_input("Student Name", placeholder="e.g., Aarav")
            student_class = st.selectbox("Class", classes, format_func=format_class_label, key="enroll_class")
            photo_src = st.radio("Photo Source", ["Upload Photo", "Browser Camera"], horizontal=True, key="enroll_src")

        with f_right:
            if photo_src == "Browser Camera":
                img_file = st.camera_input("Capture student photo", key="enroll_cam")
            else:
                img_file = st.file_uploader("Upload front-facing photo", type=["jpg", "jpeg", "png"], key="enroll_upload")

            if img_file is not None:
                st.image(img_file, caption="Preview", width=260)

        if st.button("Enroll Student", type="primary"):
            if not roll.strip() or not name.strip() or img_file is None:
                st.warning("Please enter Roll Number, Student Name, and provide a clear photo.")
            else:
                with st.spinner("Extracting FaceNet embedding and updating roster..."):
                    ok, msg = enroll_student_from_image(
                        Image.open(img_file),
                        roll=roll,
                        name=name,
                        student_class=student_class,
                    )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def main(default_page: str = "Mark Attendance") -> None:
    """Launch the FaceMask 360 Streamlit application."""
    st.set_page_config(
        page_title="FaceMask 360 · AI Attendance",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    ensure_directories()
    _init_session_state(default_page)

    students_df = load_students()
    timetable_df = load_timetable()
    page = _render_sidebar(students_df)

    if page == "Mark Attendance":
        _render_page_header(
            "Mark Attendance",
            "Real-time FaceNet recognition with automatic timetable slot matching.",
            "<span class='fm-pill fm-pill-indigo'>FaceNet · MTCNN</span>",
        )
        _render_mark_attendance(students_df, timetable_df)

    elif page == "Dashboard":
        _render_page_header(
            "Attendance Dashboard",
            "Track present and absent students, review daily logs, and export reports.",
        )
        _render_dashboard(students_df, timetable_df)

    elif page == "Timetable":
        _render_page_header(
            "Class Timetable",
            "Weekly slot schedule across classes with live active-slot detection.",
        )
        _render_timetable(timetable_df)

    elif page == "Students":
        _render_page_header(
            "Student Directory",
            "Browse enrolled students and register new faces directly into FaceNet.",
        )
        _render_students(students_df)
