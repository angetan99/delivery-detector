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

Then open `dashboard/index.html` in a browser.

## Dashboard Files

The dashboard UI lives in this folder. Backend/data files stay at the repository
root so the local app keeps working without path changes:

- `dashboard/index.html` - dashboard UI
- `main.py` - local FastAPI backend
- `model_metrics.json` - model evaluation metrics
- `retrains.json` - training version notes
- `predictions.db` - local prediction history
- `w&b_Example.png` - W&B training image used in the dashboard

## Updating The Dashboard

When you make dashboard changes, commit the moved HTML file and any related data
or backend changes together:

```bash
git add dashboard/index.html dashboard/README.md
git add main.py model_metrics.json retrains.json
git commit -m "Update dashboard"
git push origin main
```

Only add the files you actually changed.

## Links

- [GitHub Repo](https://github.com/angetan99/delivery-detector)
- [W&B Log](https://wandb.ai/angelatan2007-purdue-university/delivery-detector/workspace?nw=nwuserangelatan2007)
