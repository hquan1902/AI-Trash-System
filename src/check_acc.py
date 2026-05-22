import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.data_pipeline import ClassificationDataset
from src.model_classify import TrashNet
from torch.utils.data import DataLoader

CKPT_PATH = "models/model_classification_best.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT_PATH, map_location=device)
model = TrashNet(num_classes=ckpt["num_classes"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = ClassificationDataset(root_dir="data/classification")
loader = DataLoader(dataset, batch_size=16, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Classification Report (per class):")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)