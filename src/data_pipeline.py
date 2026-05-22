import json
import cv2
import os
import random
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from collections import defaultdict

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            folder = os.path.join(root_dir, c)
            for f in os.listdir(folder):
                self.samples.append((os.path.join(folder, f), self.class_to_idx[c]))

    def __len__(self):  # cho DataLoader biết dataset có bao nhiêu mẫu
        return len(self.samples)
    
    
    # lấy 1 ảnh → đọc, resize, normalize → trả tensor
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # OpenCV đọc BGR -> chuyển RGB cho đúng pipeline vision
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # nếu có augmentation
        if self.transform:
            img = self.transform(img)

        # nếu không dùng augmentation
        else:
            # fallback: giống val_transform
            img = cv2.resize(img, (224, 224))
            img = img.astype("float32") / 255.0
            img = torch.tensor(img).permute(2, 0, 1).float()     # {C, H, W}

        return img, label  # chuẩn hóa đầu vào để model train ổn định, nhất quán


def _clip_boxes_xyxy(boxes, w, h):
    clipped = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        if x2 > x1 and y2 > y1:
            clipped.append([x1, y1, x2, y2])
    return clipped


def _hflip_image_boxes(img, boxes):
    h, w = img.shape[:2]
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []
    for x1, y1, x2, y2 in boxes:
        nx1 = w - x2
        nx2 = w - x1
        flipped_boxes.append([nx1, y1, nx2, y2])
    return flipped_img, flipped_boxes


class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, json_path, is_train=False, hflip_prob=0.5, color_jitter_prob=0.4):
        self.img_dir = img_dir
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.image_id_to_name = {img["id"]: img["file_name"] for img in self.images}

        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        anns = self.anns_by_image.get(image_id, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(img_w), x + w)
            y2 = min(float(img_h), y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]) + 1)  # +1 vì class 0 là background trong FasterRCNN
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        # Augmentation (chỉ cho train)
        if self.is_train and len(boxes) > 0:
            if random.random() < self.hflip_prob:
                img, boxes = _hflip_image_boxes(img, boxes)
        
        if self.is_train and random.random() < self.color_jitter_prob:
            # brightness/contrast jitter nhẹ để tránh phá ảnh
            alpha = random.uniform(0.9, 1.1)   # contrast
            beta = random.uniform(-12, 12)     # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        boxes = _clip_boxes_xyxy(boxes, img.shape[1], img.shape[0])

        if len(boxes) > 0:
            synced = []
            for b, lb, ar, ic in zip(boxes, labels, areas, iscrowd):
                x1, y1, x2, y2 = b
                x1 = max(0.0, min(float(img.shape[1]), x1))
                y1 = max(0.0, min(float(img.shape[0]), y1))
                x2 = max(0.0, min(float(img.shape[1]), x2))
                y2 = max(0.0, min(float(img.shape[0]), y2))
                if x2 > x1 and y2 > y1:
                    synced.append(([x1, y1, x2, y2], lb, (x2 - x1) * (y2 - y1), ic))

            if len(synced) > 0:
                boxes, labels, areas, iscrowd = map(list, zip(*synced))
            else:
                boxes, labels, areas, iscrowd = [], [], [], []
                
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        return img_tensor, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)












class DetectionDataset(Dataset):
    def __init__(self, img_dir, json_path):
        self.img_dir = img_dir

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]:img["file_name"] for img in coco["images"]}
        self.annotations = coco["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_name = self.images[ann["image_id"]]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)

        x, y, w, h = ann["bbox"]

        crop = img[int(y):int(y + h), int(x):int(x + w)]

        crop = cv2.resize(crop, (224, 224))

        crop = crop / 255.0

        crop = torch.tensor(crop).permute(2, 0, 1).float()

        return crop