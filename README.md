# 🎯 FaceMask 360

> **Real-Time Face Recognition Attendance System powered by FaceNet & Streamlit**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/FaceNet-InceptionResnetV1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

---

## ✨ Features

- **Multi-Mode Capture**: Works seamlessly via **Browser Camera** (cloud, desktop, & mobile), **Upload Photo** (multi-face detection), and **Local Webcam** (`cv2.VideoCapture`).
- **FaceNet + MTCNN Pipeline**: Detects multiple faces per frame, computes 512-D embeddings with `InceptionResnetV1 (vggface2)`, and matches via vectorized cosine similarity.
- **Timetable-Aware Slot Matching**: Automatically detects the active slot from `timetable.csv` (`Asia/Kolkata`) with a manual slot override for testing or off-schedule sessions.
- **Live Attendance Dashboard**: View real-time Present/Absent counts, attendance rate, searchable daily & historical logs, and one-click **CSV / Excel** exports.
- **In-App Student Enrollment**: Register new students with a photo directly in the **Students** tab—automatically updating `students.csv` and `student_embeddings.pkl`.

---

## 🗂 Project Structure

```text
├── app.py                   # Primary Streamlit entrypoint
├── app2.py                  # Compatibility entrypoint
├── dashboard.py             # Dashboard entrypoint
├── facemask_app/
│   ├── config.py            # Paths, timezone, and auth configuration
│   ├── data.py              # Normalized CSV loading, timetable, attendance & exports
│   ├── recognition.py       # Cached MTCNN + FaceNet recognition & enrollment
│   └── ui.py                # Clean, responsive Streamlit interface
├── tests/
│   └── test_facemask.py     # Unit, integration, and Streamlit AppTest suite
├── students.csv             # Student roster (Roll, Name, Class)
├── timetable.csv            # Weekly class timetable
├── student_embeddings.pkl   # Precomputed FaceNet embeddings
├── requirements.txt         # Runtime dependencies
├── requirements-training.txt# Extra packages for offline CNN/MobileNet training
├── packages.txt             # System packages for Streamlit Community Cloud
└── Dockerfile               # Production container configuration
```

---

## 🚀 Running Locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Default teacher login (demo): `teacher1` / `password123` (or `admin` / `securepass`).
For production, override credentials via environment variable `FACEMASK_USERS="teacher1:yourpass,admin:youradminpass"` or Streamlit Secrets (`[auth_users]`).

---

## 🧪 Running Tests

```bash
pip install pytest
pytest -v tests/test_facemask.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t facemask360 .
docker run -p 8501:8501 -e FACEMASK_USERS="teacher1:StrongPass1" facemask360
```
