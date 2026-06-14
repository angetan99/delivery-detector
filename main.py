import sqlite3
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = "predictions.db"
METRICS_PATH = "model_metrics.json"
RETRAINS_PATH = "retrains.json" # log of retraining events (timestamp, improvements, metrics)
FRAMES_DIR = Path("inference/frames")

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
    return json.loads(path.read_text())


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