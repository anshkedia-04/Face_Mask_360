import streamlit as st
st.set_page_config(layout="centered")

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Constants
# =========================
DATASET_DIR = 'dataset'
STUDENTS_FILE = 'students.csv'
TIMETABLE_FILE = 'timetable.csv'
ATTENDANCE_DIR = 'Attendance'
EMBEDDINGS_FILE = "student_embeddings.pkl"

# =========================
# Ensure directories
# =========================
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# =========================
# Device configuration
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Load FaceNet Models (cached)
# =========================
@st.cache_resource
def load_facenet_models():
    mtcnn_model = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_model, resnet_model

mtcnn, resnet = load_facenet_models()

# =========================
# Load student embeddings
# =========================
@st.cache_resource
def load_student_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error("❌ Embeddings file not found. Please run the embedding generator script first.")
        st.stop()
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["student_embeddings"]

student_embeddings = load_student_embeddings()

# =========================
# Load Students & Timetable
# =========================
students_df = pd.read_csv(STUDENTS_FILE)   # Roll, Name, Class

# Read timetable safely and clean columns
timetable_df = pd.read_csv(TIMETABLE_FILE) # Class, Day, Slot, Start_Time, End_Time

# Normalize class and day names
timetable_df["Class"] = timetable_df["Class"].astype(str).str.strip()
timetable_df["Day"] = timetable_df["Day"].astype(str).str.strip().str.capitalize()

# Normalize time columns
timetable_df["Start_Time"] = pd.to_datetime(
    timetable_df["Start_Time"].astype(str).str.strip(), format="%H:%M", errors="coerce"
).dt.strftime("%H:%M")

timetable_df["End_Time"] = pd.to_datetime(
    timetable_df["End_Time"].astype(str).str.strip(), format="%H:%M", errors="coerce"
).dt.strftime("%H:%M")

# =========================
# Attendance utilities
# =========================
def get_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(ATTENDANCE_DIR, today)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "attendance.csv")

def load_attendance():
    file = get_attendance_file()
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        return pd.DataFrame(columns=["Roll", "Name", "Class", "Slot", "Time", "Status"])

def save_attendance(df):
    df.to_csv(get_attendance_file(), index=False)

def get_current_slot(class_name):
    today_day = datetime.now().strftime("%A").strip().capitalize()
    now_time = datetime.now().strftime("%H:%M")

    st.write(f"📅 Today: {today_day}, 🕒 Current time: {now_time}")

    # Normalize day names in timetable
    timetable_df["Day"] = timetable_df["Day"].str.strip().str.capitalize()

    # Filter timetable for the class and day
    class_timetable = timetable_df[
    timetable_df["Class"].str.strip().str.replace("Class ", "", regex=False) == class_name.strip()
]

    class_timetable = class_timetable[class_timetable["Day"] == today_day]

    st.write(f"📋 Slots for {class_name} on {today_day}:")
    st.dataframe(class_timetable)

    # Convert current time to datetime object
    now_dt = datetime.strptime(now_time, "%H:%M")

    for _, row in class_timetable.iterrows():
        try:
            start_dt = datetime.strptime(row["Start_Time"], "%H:%M")
            end_dt = datetime.strptime(row["End_Time"], "%H:%M")

            if start_dt <= now_dt <= end_dt:
                st.success(f"✅ Current Slot: {row['Slot']} ({row['Start_Time']} – {row['End_Time']})")
                return row["Slot"]

        except Exception as e:
            st.error(f"⚠️ Error parsing slot times: {e}")

    st.warning("⏳ No ongoing slot right now.")
    return None

# =========================
# Recognition & Attendance marking
# =========================
def recognize_and_mark_attendance(class_name):
    slot = get_current_slot(class_name)
    if not slot:
        st.error("⏰ No ongoing slot for this class at the current time.")
        return

    df = load_attendance()
    marked_students = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Error: Could not open camera.")
        return

    frame_placeholder = st.empty()
    st.warning(f"Class: {class_name} | Slot: {slot}")

    stop_button = st.button("⏹ Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Error receiving frame from camera.")
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]

                scores = {name: cosine_similarity([emb], [vec])[0][0] for name, vec in student_embeddings.items()}
                identified = max(scores, key=scores.get)
                score = scores[identified]

                if score >= 0.8:
                    roll = identified.split('_')[0]
                    name = '_'.join(identified.split('_')[1:])

                    student_row = students_df[students_df["Roll"].astype(str) == str(roll)]
                    if not student_row.empty:
                        if not ((df["Roll"] == roll) & (df["Slot"] == slot)).any():
                            new_record = {
                                "Roll": roll,
                                "Name": name,
                                "Class": class_name,
                                "Slot": slot,
                                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Status": "Present"
                            }
                            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
                            save_attendance(df)   # ✅ Save immediately
                            marked_students[roll] = name
                            st.success(f"✅ Marked Present: {roll} - {name}")

                        # Draw bounding box
                        boxes, _ = mtcnn.detect(img)
                        if boxes is not None:
                            box = [int(b) for b in boxes[0]]
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name} ({roll})", (box[0], box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    boxes, _ = mtcnn.detect(img)
                    if boxes is not None:
                        box = [int(b) for b in boxes[0]]
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            st.error(f"An error occurred during face detection: {e}")

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if stop_button:
            break

    cap.release()
    st.info("Attendance marking session ended.")

    st.subheader("📋 Current Attendance File")
    st.dataframe(load_attendance())   # ✅ Show saved file

    if marked_students:
        st.subheader("Attendance Marked in This Session")
        for roll, name in marked_students.items():
            st.write(f"- {roll}: {name}")
    else:
        st.warning("No new students were marked present in this session.")

# =========================
# Streamlit UI
# =========================


import streamlit as st
import pandas as pd
import os
from datetime import datetime
import time


USER_CREDENTIALS = {
    "teacher1": "password123",
    "admin": "securepass"
}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

def login():
    st.title("🔐 Teacher/Admin Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success(f"✅ Welcome {username}!")
            time.sleep(1)
            st.rerun()   # ✅ updated
        else:
            st.error("❌ Invalid Username or Password")


# ---------------- Main App ----------------
# Load data
students_df = pd.read_csv("students.csv")
timetable_df = pd.read_csv("timetable.csv")

# 🎨 Sidebar Styling
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
        color: white;
    }
    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 15px;
    }
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    div[role="radiogroup"] label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Layout
st.sidebar.markdown("<div class='sidebar-title'>🎓 FaceMask 360°</div>", unsafe_allow_html=True)

# Show current time in sidebar
current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
st.sidebar.markdown(f"<div class='sidebar-section'>🕒 <b>Current Time:</b><br>{current_time}</div>", unsafe_allow_html=True)

# Sidebar Menu
menu = st.sidebar.radio(
    "📌 Main Menu",
    ["Attendance System", "View Students Data", "View Timetable", "View Attendance", "Logout" if st.session_state["authenticated"] else "Login"]
)

# Main Title
st.title("🎓 FaceMask 360° - AI-Based Attendance System")

# ---------------- MENUS ----------------
if menu == "Attendance System":
    st.subheader("📷 Attendance System")

    # Add "All" option to dropdown
    class_options = ["All"] + list(students_df["Class"].unique())
    class_choice = st.selectbox("Select Your Class", class_options)

    if st.button("📷 Start Camera and Mark Attendance"):
        recognize_and_mark_attendance(class_choice)

elif menu == "View Students Data":
    if not st.session_state["authenticated"]:
        st.warning("🔒 Please login to access Students Data.")
        login()
    else:
        st.subheader("🧑‍🎓 Students Data")

        # Load students.csv
        students_file = "students.csv"
        if not os.path.exists(students_file) or os.stat(students_file).st_size == 0:
            st.error("⚠️ No student data found. Please register students first.")
        else:
            students_df = pd.read_csv(students_file)

            # Add class filter
            class_options = ["All"] + list(students_df["Class"].unique())
            selected_class = st.selectbox("Select a Class to view:", class_options)

            if selected_class != "All":
                filtered_df = students_df[students_df["Class"] == selected_class]
            else:
                filtered_df = students_df

            st.success(f"Showing {len(filtered_df)} student records.")
            st.dataframe(filtered_df)

elif menu == "View Timetable":
    st.subheader("📅 Class Timetable")

    class_options = ["All"] + list(timetable_df["Class"].unique())
    selected_class = st.selectbox("Select a Class timetable to view:", class_options)

    if selected_class != "All":
        filtered_df = timetable_df[timetable_df["Class"] == selected_class]
    else:
        filtered_df = timetable_df

    for day, group in filtered_df.groupby("Day"):
        st.markdown(f"### 📌 {day}")
        st.table(group[["Slot", "Start_Time", "End_Time"]].reset_index(drop=True))
        st.markdown("---")

elif menu == "View Attendance":
    if not st.session_state["authenticated"]:
        st.warning("🔒 Please login to access Attendance Records.")
        login()
    else:
        st.subheader("📊 Marked Attendance")

        today = datetime.now().strftime("%Y-%m-%d")
        attendance_path = os.path.join("attendance", today, "attendance.csv")

        if os.path.exists(attendance_path):
            attendance_df = pd.read_csv(attendance_path)

            class_options = ["All"] + list(attendance_df["Class"].unique())
            selected_class = st.selectbox("Select a Class attendance to view:", class_options)

            if selected_class != "All":
                filtered_df = attendance_df[attendance_df["Class"] == selected_class]
            else:
                filtered_df = attendance_df

            if not filtered_df.empty:
                for cls, group in filtered_df.groupby("Class"):
                    st.markdown(f"## 🏫 Class: {cls}")
                    for slot, slot_group in group.groupby("Slot"):
                        st.markdown(f"### ⏰ {slot}")
                        st.dataframe(slot_group.reset_index(drop=True))
                        st.markdown("---")
            else:
                st.warning("⚠️ No attendance records found for this class today.")
        else:
            st.error(f"❌ Attendance file not found for {today}")

elif menu == "Login":
    login()

elif menu == "Logout":
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.success("🔒 Logged out successfully!")
    time.sleep(1)
    st.rerun()   # ✅ updated

