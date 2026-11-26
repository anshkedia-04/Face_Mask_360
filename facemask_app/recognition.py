"""Face detection, FaceNet embedding matching, annotation, and enrollment."""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw, ImageFont

from facemask_app.config import DATASET_DIR, EMBEDDINGS_FILE
from facemask_app.data import normalize_class_code, upsert_student_record


def get_device() -> torch.device:
    """Select CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading FaceNet models...")
def load_facenet_models() -> tuple[MTCNN, InceptionResnetV1, torch.device]:
    """Load and cache MTCNN detector and FaceNet InceptionResnetV1."""
    device = get_device()
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet, device


@st.cache_resource(show_spinner="Loading student face embeddings...")
def load_embeddings_index(embeddings_path: str = EMBEDDINGS_FILE) -> dict[str, Any]:
    """Load student embeddings and precompute L2-normalized matrix for fast matching."""
    if not os.path.exists(embeddings_path):
        return {"keys": [], "matrix": np.empty((0, 512), dtype=np.float32), "raw": None}

    with open(embeddings_path, "rb") as f:
        raw = pickle.load(f)

    student_embeddings: dict[str, np.ndarray] = raw.get("student_embeddings", {})
    if not student_embeddings:
        return {"keys": [], "matrix": np.empty((0, 512), dtype=np.float32), "raw": raw}

    keys = list(student_embeddings.keys())
    matrix = np.stack([np.asarray(student_embeddings[k], dtype=np.float32) for k in keys])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return {"keys": keys, "matrix": matrix, "raw": raw}


def _resolve_student_metadata(key: str, students_df: pd.DataFrame | None) -> tuple[str, str, str]:
    """Resolve (roll, name, student_class) from embedding key and canonical students_df."""
    parts = key.split("_")
    roll = parts[0].strip()
    fallback_class = normalize_class_code(parts[-1]) if len(parts) >= 3 else "Unknown"
    fallback_name = "_".join(parts[1:-1]).strip() if len(parts) >= 3 else "_".join(parts[1:]).strip()

    if students_df is not None and not students_df.empty:
        match = students_df[students_df["Roll"] == roll]
        if not match.empty:
            row = match.iloc[0]
            return roll, str(row["Name"]).strip(), normalize_class_code(row["Class"])

    return roll, fallback_name or key, fallback_class


def _annotate_image(pil_img: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    """Draw crisp bounding boxes and readable pill labels onto a copy of the image."""
    annotated = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    width, height = annotated.size

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if det["matched"]:
            color = (16, 185, 129)  # Emerald
            label = f"{det['name']} ({det['roll']}) · {det['confidence'] * 100:.0f}%"
        else:
            color = (239, 68, 68)  # Rose
            label = f"Unknown · {det['confidence'] * 100:.0f}%"

        stroke = max(2, min(4, width // 220))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad_x, pad_y = 6, 4
        pill_y1 = max(0, y1 - text_h - pad_y * 2 - 2)
        pill_y2 = pill_y1 + text_h + pad_y * 2
        pill_x1 = x1
        pill_x2 = min(width, x1 + text_w + pad_x * 2)

        draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=color)
        draw.text((pill_x1 + pad_x, pill_y1 + pad_y), label, fill=(255, 255, 255), font=font)

    return annotated


def recognize_faces(
    pil_img: Image.Image,
    students_df: pd.DataFrame | None = None,
    threshold: float = 0.72,
    embeddings_path: str = EMBEDDINGS_FILE,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Detect and recognize all faces in a PIL image."""
    rgb_img = pil_img.convert("RGB")
    mtcnn, resnet, device = load_facenet_models()
    index = load_embeddings_index(embeddings_path)
    keys: list[str] = index["keys"]
    matrix: np.ndarray = index["matrix"]

    boxes, probs = mtcnn.detect(rgb_img)
    if boxes is None or len(boxes) == 0:
        return rgb_img, []

    valid_boxes = []
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.70:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            valid_boxes.append(box)

    if not valid_boxes:
        return rgb_img, []

    valid_boxes_arr = np.asarray(valid_boxes, dtype=np.float32)
    faces = mtcnn.extract(rgb_img, valid_boxes_arr, save_path=None)
    if faces is None or len(faces) == 0:
        return rgb_img, []

    if faces.ndim == 3:
        faces = faces.unsqueeze(0)

    with torch.no_grad():
        embs = resnet(faces.to(device)).cpu().numpy().astype(np.float32)

    emb_norms = np.linalg.norm(embs, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    embs = embs / emb_norms

    detections: list[dict[str, Any]] = []
    for box, emb in zip(valid_boxes_arr, embs):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        if len(keys) > 0:
            sims = emb @ matrix.T
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_key = keys[best_idx]
        else:
            best_score = 0.0
            best_key = ""

        matched = bool(best_key and best_score >= threshold)
        if matched:
            roll, name, student_class = _resolve_student_metadata(best_key, students_df)
        else:
            roll, name, student_class = "", "Unknown", ""

        detections.append(
            {
                "matched": matched,
                "roll": roll,
                "name": name,
                "student_class": student_class,
                "confidence": max(0.0, min(1.0, best_score)),
                "box": (x1, y1, x2, y2),
            }
        )

    annotated = _annotate_image(rgb_img, detections)
    return annotated, detections


def enroll_student_from_image(
    pil_img: Image.Image,
    roll: str,
    name: str,
    student_class: str,
    embeddings_path: str = EMBEDDINGS_FILE,
    students_path: str | None = None,
) -> tuple[bool, str]:
    """Detect a face in pil_img, save dataset image, update students.csv and embeddings pickle."""
    clean_roll = str(roll).strip()
    clean_name = str(name).strip().replace(" ", "")
    clean_class = normalize_class_code(student_class)

    if not clean_roll or not clean_name or not clean_class:
        return False, "Roll number, name, and class are all required."

    rgb_img = pil_img.convert("RGB")
    mtcnn, resnet, device = load_facenet_models()
    boxes, probs = mtcnn.detect(rgb_img)
    if boxes is None or len(boxes) == 0:
        return False, "No face detected in the photo. Please use a clear, front-facing photo."

    best_idx = int(np.argmax([p if p is not None else 0.0 for p in probs]))
    best_box = np.asarray([boxes[best_idx]], dtype=np.float32)
    faces = mtcnn.extract(rgb_img, best_box, save_path=None)
    if faces is None or len(faces) == 0:
        return False, "Could not extract face region from the photo."
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)

    with torch.no_grad():
        emb = resnet(faces.to(device)).cpu().numpy()[0].astype(np.float32)

    key = f"{clean_roll}_{clean_name}_{clean_class}"
    person_dir = os.path.join(DATASET_DIR, key)
    os.makedirs(person_dir, exist_ok=True)
    existing_imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_idx = len(existing_imgs) + 1
    rgb_img.save(os.path.join(person_dir, f"{next_idx}.jpg"), format="JPEG")

    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        X = list(data.get("X", []))
        y = list(data.get("y", []))
        student_embeddings = dict(data.get("student_embeddings", {}))
    else:
        X, y, student_embeddings = [], [], {}

    # Remove stale keys for the same roll number if updating an existing student
    stale_keys = [k for k in student_embeddings if k.split("_")[0] == clean_roll and k != key]
    for sk in stale_keys:
        student_embeddings.pop(sk, None)

    X.append(emb)
    y.append(key)
    person_embs = [np.asarray(vec, dtype=np.float32) for vec, lbl in zip(X, y) if lbl == key]
    student_embeddings[key] = np.mean(person_embs, axis=0)

    with open(embeddings_path, "wb") as f:
        pickle.dump({"X": np.asarray(X), "y": np.asarray(y), "student_embeddings": student_embeddings}, f)

    if students_path:
        upsert_student_record(clean_roll, clean_name, clean_class, path=students_path)
    else:
        upsert_student_record(clean_roll, clean_name, clean_class)
    load_embeddings_index.clear()
    return True, f"Enrolled {clean_name} (Roll {clean_roll}, Class {clean_class}) and updated FaceNet embeddings."
