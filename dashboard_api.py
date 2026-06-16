import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "predictions.db"
METRICS_PATH = "model_metrics.json"
RETRAINS_PATH = "retrains.json" # log of retraining events (timestamp, improvements, metrics)
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "inference" / "frames"
DASHBOARD_ASSETS_DIR = BASE_DIR
LOCAL_TZ = ZoneInfo(os.getenv("DASHBOARD_TIMEZONE", "America/Los_Angeles"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/predictions")
def get_predictions(limit: int = 50, offset: int = 0):
    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


class LabelUpdate(BaseModel):
    correct: int

@app.patch("/predictions/{pred_id}/label")
def label_prediction(pred_id: int, body: LabelUpdate):
    if body.correct not in (0, 1):
        raise HTTPException(400, "correct must be 0 or 1")
    conn = get_db()
    cur = conn.execute(
        "UPDATE predictions SET correct = ? WHERE id = ?", (body.correct, pred_id)
    )
    conn.commit()
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "prediction not found")
    conn.close()
    return {"id": pred_id, "correct": body.correct}


@app.get("/predictions/timeline")
def get_timeline():
    conn = get_db()
    cur = conn.execute("""
        SELECT
            date(timestamp) AS day,
            COUNT(*) AS total_labeled,
            SUM(correct) AS num_correct
        FROM predictions
        WHERE correct IS NOT NULL
        GROUP BY day
        ORDER BY day ASC
    """)
    rows = []
    for r in cur.fetchall():
        total = r["total_labeled"]
        rows.append({
            "day": r["day"],
            "total_labeled": total,
            "accuracy": r["num_correct"] / total if total else None,
        })
    conn.close()
    return rows


def parse_dashboard_timestamp(value: str):
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, "since must be an ISO timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=LOCAL_TZ)
    return timestamp.astimezone(timezone.utc)


@app.get("/predictions/by-hour")
def get_predictions_by_hour(since: str | None = None):
    since_utc = parse_dashboard_timestamp(since) if since else None
    conn = get_db()
    cur = conn.execute("""
        SELECT timestamp, correct
        FROM predictions
        WHERE correct IS NOT NULL
    """)
    buckets = {
        hour: {"hour": hour, "total_labeled": 0, "num_correct": 0}
        for hour in range(24)
    }
    for r in cur.fetchall():
        timestamp = datetime.fromisoformat(r["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if since_utc and timestamp < since_utc:
            continue
        local_hour = timestamp.astimezone(LOCAL_TZ).hour
        buckets[local_hour]["total_labeled"] += 1
        buckets[local_hour]["num_correct"] += r["correct"]
    conn.close()

    rows = []
    for bucket in buckets.values():
        total = bucket["total_labeled"]
        rows.append({
            "hour": bucket["hour"],
            "total_labeled": total,
            "accuracy": bucket["num_correct"] / total if total else None,
        })
    return rows


@app.get("/retrains")
def get_retrains():
    path = Path(RETRAINS_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text())


@app.get("/metrics")
def get_metrics():
    path = Path(METRICS_PATH)
    if not path.exists():
        return {"error": "model_metrics.json not found"}
    metrics = json.loads(path.read_text())
    wandb_run_url = os.getenv("WANDB_RUN_URL")
    if wandb_run_url:
        metrics.setdefault("wandb", {})["run_url"] = wandb_run_url
    return metrics


@app.get("/assets/{filename}")
def get_dashboard_asset(filename: str):
    path = DASHBOARD_ASSETS_DIR / Path(filename).name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(path)


@app.get("/examples")
def get_examples(low: float = 0.6, high: float = 0.9, n: int = 5):
    conn = get_db()
    buckets = {
        "low": (0, low),
        "medium": (low, high),
        "high": (high, 1.0),
    }
    result = {}
    for name, (lo, hi) in buckets.items():
        cur = conn.execute("""
            SELECT * FROM predictions
            WHERE mean_prob >= ? AND mean_prob < ?
            ORDER BY timestamp DESC LIMIT ?
        """, (lo, hi, n))
        result[name] = [dict(r) for r in cur.fetchall()]
    conn.close()
    return result


@app.get("/frames/{filename}")
def get_frame(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(path)


@app.get("/frames-for/{clip_filename}")
def get_frame_for_clip(clip_filename: str):
    """Resolve any frame image belonging to a clip, e.g. clip_filename
    '20260612_160845' matches 'inference/frames/20260612_160845_frame2.jpeg'
    regardless of the frame number suffix."""
    stem = Path(clip_filename).stem
    matches = sorted(FRAMES_DIR.glob(f"{stem}_frame*.jpeg"))
    if not matches:
        raise HTTPException(404, "no frame found for clip")
    return {"filename": matches[0].name}
