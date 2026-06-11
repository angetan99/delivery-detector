import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

# same transforms as training val set
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load val data
data_dir = os.path.join(os.path.dirname(__file__), "data", "val")
dataset = datasets.ImageFolder(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
class_names = dataset.classes

# load model
device = torch.device("cpu")
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'delivery_detector.pt'), map_location=device))
model.eval()

# run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# print results
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print()
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))