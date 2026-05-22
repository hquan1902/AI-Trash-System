import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_pipeline import ClassificationDataset, train_transform, val_transform
from src.model_classify import TrashNet



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_val_sets(root_dir, train_ratio=0.8, seed=42):
    base_dataset = ClassificationDataset(root_dir=root_dir, transform=None)
    n = len(base_dataset)
    train_size = int(train_ratio * n)

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    # 2 dataset khác transform nhưng chung class_to_idx
    train_dataset = ClassificationDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = ClassificationDataset(root_dir=root_dir, transform=val_transform)

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)

    return base_dataset, train_set, val_set, train_idx


def compute_class_weights(samples, train_idx, num_classes):
    labels = [samples[i][1] for i in train_idx]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)

    # inverse frequency
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


@torch.no_grad()  # tắt gradient khi evaluate để nhanh và tiết kiệm RAM
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / max(1, len(loader))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1, y_true, y_pred


def main():
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    #  data
    base_dataset, train_set, val_set, train_idx = build_train_val_sets(
        root_dir="data/classification",
        train_ratio=0.8,
        seed=42
    )

    num_classes = len(base_dataset.class_to_idx)

    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(base_dataset)}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    class_weights, class_counts = compute_class_weights(
        samples=base_dataset.samples,
        train_idx=train_idx,
        num_classes=num_classes
    )
    print("Train class counts:", class_counts.tolist())
    print("Class weights:", [round(x, 4) for x in class_weights.tolist()])


    batch_size = 16
    cpu_count = os.cpu_count() or 0
    num_workers = min(4, cpu_count)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Batch size: {batch_size}")
    print(f"Train steps/epoch: {len(train_loader)}")
    print(f"Val steps/epoch: {len(val_loader)}")

    # model (from scratch)
    model = TrashNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",      # theo val_f1
        factor=0.5,
        patience=2
    )

    epochs = 30
    best_val_f1 = -1.0
    best_val_acc = 0.0
    early_stop_patience = 6
    no_improve_epochs = 0
    best_y_true, best_y_pred = None, None


    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={current_lr:.6f} | "
            f"time={epoch_time:.1f}s"
        )

        # Save LAST mỗi epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": base_dataset.class_to_idx,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": current_lr,
                "class_counts": class_counts.tolist(),
            },
            "models/model_classification_last.pth",
        )

        # Save BEST theo val_acc
        # Save BEST theo val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            no_improve_epochs = 0
            best_y_true = list(y_true)
            best_y_pred = list(y_pred)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": base_dataset.class_to_idx,
                    "num_classes": num_classes,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": current_lr,
                    "class_counts": class_counts.tolist(),
                },
                "models/model_classification_best.pth",
            )
            print(f"  -> New BEST saved (val_f1={best_val_f1:.4f}, val_acc={best_val_acc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"  -> No improve epochs: {no_improve_epochs}/{early_stop_patience}")

        if no_improve_epochs >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")
    print("Saved: models/model_classification_last.pth")
    print("Saved: models/model_classification_best.pth")

    if best_y_true is not None and best_y_pred is not None:
        y_true, y_pred = best_y_true, best_y_pred

    # Tạo confusion matrix từ kết quả val epoch cuối
    cm = confusion_matrix(y_true, y_pred)
    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True") 
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    plt.close()

    print("Saved: models/confusion_matrix.png")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))  # in report precision/recall/f1


if __name__ == "__main__":
    main()



