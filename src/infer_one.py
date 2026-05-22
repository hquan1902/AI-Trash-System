import cv2
import torch
import numpy as np
from src.model_classify import TrashNet


CKPT_PATH = "models/model_classification_best.pth"
IMG_PATH = "data/classification/trash/trash_22.jpg"


def preprocess(img_bgr):
    """Đọc ảnh BGR → normalize → tensor [1,3,224,224]"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = ckpt["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Load model
    model = TrashNet(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read image
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {IMG_PATH}")

    x = preprocess(img).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    conf = conf.item()
    pred_name = idx_to_class[pred_idx]

    print(f"\nImage: {IMG_PATH}")
    print(f"Predict: {pred_name} (class_id={pred_idx})")
    print(f"Confidence: {conf:.4f}")
    print(f"\nAll class probs:")
    for i in range(num_classes):
        print(f"  {idx_to_class[i]}: {probs[0, i].item():.4f}")


if __name__ == "__main__":
    main()