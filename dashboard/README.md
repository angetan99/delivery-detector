# Dashboard

The Delivery Detector dashboard is a local monitoring view for model quality and
live prediction history. It shows model metrics, a confusion matrix, W&B training
context, confidence examples, prediction history, and frame previews for logged
clips.

## What It Shows

- Current model metrics: accuracy, precision, recall, and F1
- Confusion matrix for the latest evaluated model
- Most recent training run context
- Accuracy over time from manually labeled live predictions
- Example predictions by confidence level
- Prediction log with frame previews and correctness labels

## Run Locally

Start the FastAPI backend from the repository root:

```bash
uvicorn main:app --reload
```

Then open `index-4.html` in a browser.

## Dashboard Files

The dashboard files currently live at the repository root so the local app keeps
working without path changes:

- `index-4.html` - dashboard UI
- `main.py` - local FastAPI backend
- `model_metrics.json` - model evaluation metrics
- `retrains.json` - training version notes
- `predictions.db` - local prediction history
- `w&b_Example.png` - W&B training image used in the dashboard

## Links

- [GitHub Repo](https://github.com/angetan99/delivery-detector)
- [W&B Log](https://wandb.ai/angelatan2007-purdue-university/delivery-detector/workspace?nw=nwuserangelatan2007)
