import json
import logging
import os
import sqlite3
import time
from datetime import datetime

import cv2
import pyaarlo
import torch
import torch.nn as nn
from PIL import Image
from pyaarlo.constant import MOTION_DETECTED_KEY
from torchvision import models, transforms

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ARLO_USERNAME = os.environ["ARLO_USERNAME"]
ARLO_PASSWORD = os.environ["ARLO_PASSWORD"]
CAMERA_NAME = os.environ["CAMERA_NAME"]
MODEL_PATH         = "delivery_detector.pt"
DB_PATH            = "predictions.db"
CLIPS_DIR          = os.path.join("inference", "clips")
FRAMES_DIR         = os.path.join("inference", "frames")

CLIP_OFFSET_SEC    = 3      # seconds into clip before first frame
NUM_FRAMES         = 6      # frames to sample
CLIP_WAIT_SEC      = 20     # time to wait for Arlo to finish uploading
DELIVERY_THRESHOLD = 0.5

DELIVERY_HOUR_START = 6   # 6am
DELIVERY_HOUR_END   = 21  # 9pm

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────

TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    log.info(f"Model loaded from {path}")
    return model


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(clip_path, timestamp_str):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        log.error(f"Could not open clip: {clip_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    log.info(f"Clip: {duration_sec:.1f}s @ {fps:.1f}fps")

    pil_frames = []
    sample_end = duration_sec - 10
    stride = (sample_end - CLIP_OFFSET_SEC) / (NUM_FRAMES - 1)
    for i in range(NUM_FRAMES):
        target_sec = CLIP_OFFSET_SEC + i * stride
        if target_sec >= sample_end:
            log.warning(f"Frame {i} at {target_sec:.1f}s exceeds clip length, stopping early.")
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
        ret, bgr = cap.read()
        if not ret:
            log.warning(f"Could not read frame at {target_sec:.1f}s.")
            continue
        frame_filename = f"{timestamp_str}_frame{i}.jpeg"
        frame_path = os.path.join(FRAMES_DIR, frame_filename)
        cv2.imwrite(frame_path, bgr)
        log.info(f"Saved frame {i} → {frame_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))

    cap.release()
    return pil_frames


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, frames):
    softmax = nn.Softmax(dim=1)
    delivery_probs = []

    with torch.no_grad():
        for i, img in enumerate(frames):
            tensor = TRANSFORMS(img).unsqueeze(0)
            probs = softmax(model(tensor))[0]
            prob = probs[0].item()  # index 0 = delivery
            delivery_probs.append(prob)
            log.info(f"Frame {i}: delivery prob = {prob:.3f}")

    mean_prob = sum(delivery_probs) / len(delivery_probs)
    prediction = "delivery" if mean_prob >= DELIVERY_THRESHOLD else "no_delivery"
    confident = abs(mean_prob - DELIVERY_THRESHOLD) > 0.2
    log.info(f"Mean prob: {mean_prob:.3f} → {prediction}")

    return {
        "frame_probs": delivery_probs,
        "mean_prob":   mean_prob,
        "prediction":  prediction,
        "confident":   confident,
    }


# ── SQLite ────────────────────────────────────────────────────────────────────

def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            prediction    TEXT    NOT NULL,
            mean_prob     REAL    NOT NULL,
            frame_probs   TEXT    NOT NULL,
            num_frames    INTEGER NOT NULL,
            clip_filename TEXT    NOT NULL,
            confident     INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_prediction(conn, result, clip_filename):
    conn.execute("""
        INSERT INTO predictions
            (timestamp, prediction, mean_prob, frame_probs, num_frames, clip_filename, confident)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        result["prediction"],
        result["mean_prob"],
        json.dumps(result["frame_probs"]),
        len(result["frame_probs"]),
        clip_filename,
        int(result["confident"]),
    ))
    conn.commit()
    log.info(f"Logged to DB: {result['prediction']} (mean_prob={result['mean_prob']:.3f})")


# ── Motion handler ────────────────────────────────────────────────────────────

def on_motion(model, conn, camera, attr, value):
    current_hour = datetime.now().hour  # local time
    if not value:
        return
    if not (DELIVERY_HOUR_START <= current_hour <= DELIVERY_HOUR_END):
        log.info("Motion outside daylight hours — skipping.")
        return

    log.info("Motion detected.")
    log.info(f"Waiting {CLIP_WAIT_SEC}s for clip to upload…")
    time.sleep(CLIP_WAIT_SEC)

    camera.update_media(wait=True)
    video = camera.last_video
    if video is None:
        log.warning("No video found after motion — skipping.")
        return

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    clip_filename = f"{timestamp_str}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_filename)

    success = video.download_video(filename=clip_path)
    if not success:
        log.error("Failed to download clip — skipping.")
        return
    log.info(f"Clip saved → {clip_path}")

    frames = extract_frames(clip_path, timestamp_str)
    if not frames:
        log.warning("No frames extracted — skipping inference.")
        return

    result = run_inference(model, frames)
    log_prediction(conn, result, clip_filename)

    emoji = "📦" if result["prediction"] == "delivery" else "🚶"
    print(f"\n{emoji}  {result['prediction'].upper()}  "
          f"(mean prob: {result['mean_prob']:.1%})  "
          f"{'[confident]' if result['confident'] else '[uncertain]'}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)
    conn = init_db(DB_PATH)

    log.info("Connecting to Arlo…")
    arlo = pyaarlo.PyArlo(
        username=ARLO_USERNAME,
        password=ARLO_PASSWORD,
        tfa_type="email",
        tfa_source="console",
        synchronous_mode=False,
    )

    log.info(f"Doorbells: {[d.name for d in arlo.doorbells]}")
    log.info(f"Cameras: {[d.name for d in arlo.cameras]}")

    camera = arlo.lookup_camera_by_name(CAMERA_NAME)
    if camera is None:
        log.error(f"Device '{CAMERA_NAME}' not found. Check name matches the Arlo app exactly.")
        arlo.stop()
        return

    log.info(f"Found device: {camera.name}")
    camera.add_attr_callback(
        MOTION_DETECTED_KEY,
        lambda attr, value: on_motion(model, conn, camera, attr, value),
    )

    log.info("Listening for motion… (Ctrl-C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        arlo.stop(logout=True)
        conn.close()


if __name__ == "__main__":
    main()