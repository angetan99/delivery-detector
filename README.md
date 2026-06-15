# Delivery Detector

A real-time computer vision system that monitors an Arlo doorbell camera, detects delivery drivers using a fine-tuned ResNet50 model, and sends an instant push notification to my phone with a snapshot from the clip.

Built end-to-end: data collection, model training, live inference pipeline, and hardware integration.

![Delivery Detector Dashboard](dashboard/assets/Delivery%20Detector%20Dashboard.png)

---

## How It Works

When the Arlo camera detects motion, the script downloads the clip, samples frames at regular intervals, and runs each frame through the classifier. If the average delivery probability clears the threshold, I recieve a Pushover notification with a confidence score and a snapshot from the clip attached.

```
Arlo motion event
  → download clip
  → extract frames
  → ResNet50 inference on each frame
  → average probabilities
  → log to SQLite + notify phone if delivery
This happens within 20 seconds of the person at the door leaving, so it's very quick and real-time.
```

---

## Model

- **Architecture:** ResNet50 with fine-tuned classification head (transfer learning via PyTorch)
- **Dataset:** (Initial training) 736 frames collected from real doorbell footage — 366 delivery, 370 no-delivery
- **Training:** 80/20 train/val split, trained on CPU
- **Validation accuracy:** 98.65%
- **Recall:** 1.0 — zero false negatives on the validation set
- **Experiment tracking:** Weights & Biases

> **Real-world results:** *(will fill in after summer data collection)*

---

## Stack

- Python, PyTorch, torchvision
- OpenCV for frame extraction
- pyaarlo for Arlo camera integration
- Pushover for phone notifications
- SQLite for prediction logging
- scikit-learn for evaluation metrics
- Weights & Biases for experiment tracking
- launchd for keeping the script running unattended on macOS

---

## Project Structure

```
detect_deliveries.py   # Motion listener, inference pipeline, notification
train_model.py         # Fine-tune ResNet50 on custom dataset
evaluate_model.py      # Confusion matrix, precision, recall, F1
data/                  # Training and validation image folders
  train/
    delivery/
    no_delivery/
  val/
    delivery/
    no_delivery/
inference/
  clips/               # Downloaded .mp4s from Arlo
  frames/              # Extracted frames for inference
delivery_detector.pt   # Trained model weights (held locally)
predictions.db         # SQLite prediction log
```

---

## Setup

1. Clone the repo and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python-headless pillow pyaarlo python-dotenv requests scikit-learn wandb
```

2. Copy `.env.example` to `.env` and fill in your credentials:

```env
ARLO_USERNAME=
ARLO_PASSWORD=
CAMERA_NAME=
PUSHOVER_USER_KEY=
PUSHOVER_API_TOKEN=
```

3. Place `delivery_detector.pt` in the project root (not committed to git, large file).

---

## Usage

Run the live detector, which will continuously listen for motion in the background:

```bash
python detect_deliveries.py
```

Train a new model, which is done periodically as more camera data comes in:

```bash
python train_model.py
```

Evaluate the current model:

```bash
python evaluate_model.py
```

---

## Notes

- Motion events are only processed between 6am–9pm local time (model trained on daytime footage)
- The last ~13 seconds of Arlo clips are typically empty and could be removed based on your unique circumstance
- Prediction history is logged to `predictions.db` with timestamp, mean confidence, and individual frame probabilities
- `delivery_detector.pt` is excluded from git because large binary file
