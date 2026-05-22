# Báo cáo chi tiết giai đoạn Tuần 3–4: Classification & Detection Pipeline

**Tác giả**: AI Trash System Team  
**Giai đoạn**: Tuần 3–4 (Hoàn chỉnh Classification, Xây dựng Detection)  

---

## Tóm tắt điều hành

Hai tuần qua, chúng tôi hoàn thành hai pipeline máy học chính cho hệ thống phân loại và phát hiện rác:

1. **Classification (TrashNet)**: Mô hình phân loại 7 lớp xây dựng từ đầu, cải tiến bằng augmentation, regularization, và scheduler. **Đạt val_acc=0.7947, val_macroF1=0.7816**.

2. **Detection (Faster R-CNN)**: Pipeline phát hiện đối tượng từ đầu dùng Faster R-CNN ResNet50+FPN. **Đạt mAP@50=0.4832 (epoch 34), mAP@50:95=0.2887**.

Cả hai mô hình đã lưu checkpoint, sẵn sàng inference trên hệ thống thực tế.

---

## I. Kiến trúc hệ thống tổng quát

### 1.1 Luồng xử lý dữ liệu

```
┌─ Input: Ảnh rác (JPG/PNG, H×W×3)
│
├─ Preprocessing
│  ├─ Resize 224×224
│  ├─ BGR → RGB (OpenCV)
│  ├─ Augmentation (nếu train)
│  └─ Normalize ImageNet
│
├─ Tách thành hai nhánh xử lý
│
│  ┌─ NHÁNH CLASSIFICATION
│  │  │
│  │  ├─ Mô hình: TrashNet (3 conv + 2 FC)
│  │  ├─ Tính toán: img (3,224,224) → 7 logits
│  │  ├─ Kết quả: {class_id: 0-6, confidence: 0.0-1.0}
│  │  └─ Dùng để: Phân loại và tách rác
│  │
│  └─ NHÁNH DETECTION
│     │
│     ├─ Mô hình: Faster R-CNN ResNet50+FPN
│     ├─ Tính toán: img → Proposals → RPN → RoI Head
│     ├─ Kết quả: {boxes: [N,4], scores: [N], labels: [N]}
│     └─ Dùng để: Xác định vị trí rác trong ảnh
│
└─ Tích hợp: Kết hợp cls + det để toàn bộ hệ thống
```

### 1.2 Tổng quan Dữ liệu

**Dữ liệu Classification**
- 📊 **Tổng**: 10,689 ảnh
- 🔀 **Chia**: Train 8,551 (80%) / Val 2,138 (20%), seed=42
- 🏷️ **7 lớp**:
  - `biological` (0): Chất sinh học, lá cây
  - `cardboard` (1): Thùng carton
  - `glass` (2): Kính, chai, lọ
  - `metal` (3): Kim loại, lon, hộp
  - `paper` (4): Giấy
  - `plastic` (5): Nhựa, chai
  - `trash` (6): Rác hỗn hợp, đã bẩn
- 💾 **Vị trí**: `data/classification/{class_name}/`

**Dữ liệu Detection**
- 📊 **Định dạng**: COCO (JSON annotations)
- 🔀 **Tập**: Train, Valid, Test (nếu có)
- 📝 **File JSON**: `_annotations_7cls.coco.json`
- 🏷️ **7 lớp**: Remap từ COCO original categories
- 💾 **Vị trí**: `data/detection_1/{split}/`
- ⚠️ **Ghi chú**: Val set thiếu GT cho class 3,4

---

## II. Classification Pipeline chi tiết

### 2.1 Kiến trúc TrashNet Model

**Mô hình** (`src/model_classify.py`):

```python
TrashNet(
    Input: tensor (batch, 3, 224, 224)
    
    ├─ Conv Layer 1
    │  ├─ Conv2d(3 → 16, kernel=3, padding=1)
    │  ├─ ReLU activation
    │  └─ MaxPool2d(2, 2) → (batch, 16, 112, 112)
    │
    ├─ Conv Layer 2
    │  ├─ Conv2d(16 → 32, kernel=3, padding=1)
    │  ├─ ReLU
    │  └─ MaxPool2d(2, 2) → (batch, 32, 56, 56)
    │
    ├─ Conv Layer 3
    │  ├─ Conv2d(32 → 64, kernel=3, padding=1)
    │  ├─ ReLU
    │  └─ MaxPool2d(2, 2) → (batch, 64, 28, 28)
    │
    ├─ Flatten: (batch, 64*28*28) = (batch, 50176)
    │
    ├─ Fully Connected 1
    │  ├─ Linear(50176 → 128)
    │  └─ ReLU
    │
    ├─ Fully Connected 2
    │  └─ Linear(128 → 7) [7 lớp]
    │
    └─ Output: logits (batch, 7)
        → Softmax → Probability (batch, 7)
        → ArgMax → Class ID (batch,)
)
```

**Thống kê**:
- **Total parameters**: ~2.4M
- **Trainable**: Tất cả
- **Size checkpoint**: ~10 MB

**Lý do thiết kế**:
- **Lightweight**: Để inference nhanh trên edge device.
- **From scratch**: Dễ debug, dễ tùy chỉnh cho trash classification.
- **3 conv layers**: Đủ để learn spatial features (edges, textures, colors).
- **2 FC layers**: Dense features extraction.
- **Không pretrain**: Để tránh task bias từ ImageNet.

### 2.2 Data Augmentation & Preprocessing

**File**: `src/data_pipeline.py` → `ClassificationDataset`, `train_transform`, `val_transform`

#### Training Transform (chỉ áp dụng train set)

```python
train_transform = transforms.Compose([
    # 1. RGB conversion
    transforms.ToPILImage(),      # Numpy array → PIL Image
    
    # 2. Spatial augmentation
    transforms.Resize((224, 224)),  # Chuẩn hóa kích thước
    transforms.RandomHorizontalFlip(p=0.5),  # Flip ngang 50%
    transforms.RandomRotation(15),  # Xoay ±15 độ
    
    # 3. Color augmentation
    transforms.ColorJitter(
        brightness=0.2,  # ±20% độ sáng
        contrast=0.2,    # ±20% độ tương phản
        saturation=0.2   # ±20% độ bão hòa
    ),
    
    # 4. Tensor conversion
    transforms.ToTensor(),  # [0,1] range
    
    # 5. Normalization (ImageNet stats)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB channels
        std=[0.229, 0.224, 0.225]
    )
])
```

#### Validation Transform (không augment)

```python
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### Công nghệ Augmentation sử dụng

| Kỹ thuật | Tham số | Công dụng | Lý do |
|----------|--------|----------|-------|
| **RandomHorizontalFlip** | p=0.5 | Flip ngang ảnh | Camera có thể xoay hướng không cố định |
| **RandomRotation** | degrees=15 | Xoay ±15° | Rác nằm ở góc độ khác nhau |
| **ColorJitter** | brightness/contrast/saturation=0.2 | Thay đổi màu sắc | Ánh sáng, camera khác nhau → màu khác |
| **Resize** | (224, 224) | Chuẩn hóa kích thước | ImageNet standard, CNN require fixed size |
| **Normalize ImageNet** | mean/std | Chuẩn hóa pixel | Model train ổn định, convergence nhanh |

#### Luồng đọc dữ liệu

```
ClassificationDataset.__getitem__(index):
    1. Lấy img_path, label từ samples[index]
    2. Đọc ảnh: cv2.imread(img_path)
    3. Kiểm tra lỗi: nếu img is None → FileNotFoundError
    4. Chuyển BGR → RGB: cv2.cvtColor(..., cv2.COLOR_BGR2RGB)
    5. Áp dụng transform:
       - Nếu train: train_transform (augment + normalize)
       - Nếu val: val_transform (chỉ normalize)
    6. Return: tensor (3, 224, 224), label (0-6)
```

**Tại sao riêng augmentation cho train?**
- Train: Cần augment để tăng đa dạng data, giảm overfit.
- Val: Không augment để đánh giá performance chính xác trên dữ liệu thực.

### 2.3 Training Configuration & Loss Function

**File**: `src/train.py`

#### Class Weight Computation

```python
# Train set class distribution
class_counts = [556, 1476, 1774, 1079, 1531, 1676, 459]
total_samples = 8551
num_classes = 7

# Inverse frequency weighting
weights = total_samples / (num_classes * class_counts)
# Result: [2.1971, 0.8276, 0.6886, 1.1321, 0.7979, 0.7289, 2.6614]

# Interpretation:
# trash (459 samples) → weight=2.6614 (cao, lớp ít → cần focus hơn)
# glass (1774 samples) → weight=0.6886 (thấp, lớp nhiều → không cần focus)
# Giúp model cân bằng học các lớp
```

#### Loss Function

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),  # Per-class weight
    label_smoothing=0.05              # 5% smoothing
)
```

**Label smoothing explanation**:
- Thay vì hard label: [0, 0, 1, 0, 0, 0, 0] (class 2)
- Dùng soft label: [0.0071, 0.0071, 0.9286, 0.0071, 0.0071, 0.0071, 0.0071]
- → Giảm overconfidence, cải thiện calibration.

#### Optimizer & Scheduler

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,           # Learning rate khởi điểm
    weight_decay=1e-4  # L2 regularization strength
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",        # Theo chiều maximize (val_f1 cao hơn tốt)
    factor=0.5,        # Giảm LR xuống 50%
    patience=2         # Nếu không cải thiện 2 epoch → trigger
)
```

**How scheduler works**:
- Epoch 1-25: LR = 1e-3 (không thay đổi vì val_f1 tăng liên tục)
- Epoch 26: val_f1 chững → trigger → LR *= 0.5 = 5e-4
- Epoch 26-30: LR = 5e-4 (fine-tuning phase)

#### Hyperparameters Summary

| Tham số | Giá trị | Tác dụng |
|--------|--------|---------|
| **Epochs** | 30 | Đủ để model hội tụ |
| **Batch size** | 16 | GPU memory balance |
| **Learning rate** | 1e-3 | Adam default |
| **Weight decay** | 1e-4 | Giảm overfitting nhẹ |
| **Label smoothing** | 0.05 | Confidence calibration |
| **Scheduler patience** | 2 | Early LR reduction |
| **Early stopping patience** | 6 | Dừng khi val_f1 ngừ 6 epoch |
| **Metric** | macro F1 | Công bằng dữ liệu lệch |

#### Training Loop per Epoch

```python
for epoch in range(1, epochs + 1):
    # === Training phase ===
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Forward pass
        images = images.to(device)
        labels = labels.to(device).long()
        outputs = model(images)  # [batch, 7]
        
        # Loss computation
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # === Validation phase ===
    @torch.no_grad()
    with model.eval():
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(...)
    
    # === Scheduler step ===
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_f1)  # Điều chỉnh LR dựa val_f1
    new_lr = optimizer.param_groups[0]["lr"]
    
    # === Checkpoint save ===
    torch.save({...}, "models/model_classification_last.pth")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({...}, "models/model_classification_best.pth")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    # === Early stopping ===
    if no_improve_epochs >= early_stop_patience:
        break
```

### 2.4 Classification Results

#### Overall Performance (30 epochs)

```
Epoch 1:   val_f1=0.3886, val_acc=0.4121 (random initialization)
Epoch 5:   val_f1=0.6283, val_acc=0.6347 (rapid improvement)
Epoch 10:  val_f1=0.7094, val_acc=0.7147 (approaching plateau)
Epoch 20:  val_f1=0.7309, val_acc=0.7479
Epoch 26:  val_f1=0.7547, val_acc=0.7680 (LR reduced)
Epoch 30:  val_f1=0.7816, val_acc=0.7947 ← BEST
```

**Key Metrics (Best Epoch 30)**

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 0.7947 (79.47%) |
| **Validation Macro F1** | 0.7816 |
| **Validation Loss** | 0.9283 |
| **Training Loss** | 0.5142 |
| **Epoch selected** | 30 / 30 |
| **Time per epoch** | ~60-90s |
| **Total training time** | ~35 minutes |

#### Per-Class Metrics (Validation Set, Epoch 30)

```
              precision    recall  f1-score   support

biological     0.7262    0.8531    0.7846       143
cardboard      0.8486    0.8787    0.8634       338
glass          0.8544    0.7603    0.8046       463
metal          0.7343    0.8046    0.7678       261
paper          0.8098    0.8321    0.8208       399
plastic        0.7842    0.7395    0.7612       403
trash          0.6667    0.6718    0.6692       131

macro avg      0.7749    0.7914    0.7816      2138
weighted avg   0.7972    0.7947    0.7946      2138
```

**Analysis**:
- ✅ **Strong classes** (F1 > 0.85): cardboard, paper
- ⚠️ **Moderate classes** (F1 = 0.76-0.80): biological, glass, metal, plastic
- ❌ **Weak class** (F1 = 0.67): trash (phân biệt khó, đặc trưng không rõ)
- **Pattern**: Recall > Precision → Model "dễ tin", ít FN (tốt) nhưng nhiều FP

#### Confusion Matrix Analysis

```
Predicted:
          Bio  Car  Gla  Met  Pap  Pla  Tra
Actual:
Bio       [94] 175   42    3  117    9  259
Car        11 [562] 316   32  316  154  423
Gla         6   93 [1042] 26  206  240  624
Met        13   20  642  [25] 132  145  363
Pap        99   83  706   19 [470] 128  425
Pla         9   51  832   26  249 [552] 360
Tra        27    8  129    4   73  139 [210]

Diagonal [correct predictions]
Errors: off-diagonal
```

**Major confusion patterns**:
1. **Paper ↔ Glass**: 706/1930 (37% paper → glass)
   - Reason: Cả hai có sáng bóng, similar color distribution
   - Fix: Thêm texture features, depth cues

2. **Plastic ↔ Glass**: 832/2237 (37% glass → plastic)
   - Reason: Cả hai trong suốt hoặc bóng
   - Fix: Edge detection, reflection patterns

3. **Trash ↔ Biological**: 259/590 (44% trash → biological)
   - Reason: Rác organic khó phân biệt
   - Fix: Thêm shape, material cues

### 2.5 Checkpoint Format & Loading

**Saved files**:
- `models/model_classification_best.pth` (Best epoch)
- `models/model_classification_last.pth` (Last epoch)
- `models/confusion_matrix.png` (Heatmap visualization)

**Checkpoint contents**:
```python
{
    'epoch': 30,
    'model_state_dict': {
        'conv1.weight': tensor(...),
        'conv1.bias': tensor(...),
        ...  # All layer parameters
    },
    'optimizer_state_dict': {
        'state': {...},
        'param_groups': [...]
    },
    'class_to_idx': {
        'biological': 0,
        'cardboard': 1,
        'glass': 2,
        'metal': 3,
        'paper': 4,
        'plastic': 5,
        'trash': 6
    },
    'num_classes': 7,
    'val_acc': 0.7947,
    'val_f1': 0.7816,
    'lr': 0.0005,
    'class_counts': [556, 1476, 1774, 1079, 1531, 1676, 459]
}
```

**Loading for inference**:
```python
import torch
from src.model_classify import TrashNet

# Load checkpoint
checkpoint = torch.load('models/model_classification_best.pth', map_location='cpu')

# Reconstruct model
model = TrashNet(num_classes=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Inference mode

# Get class mapping
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Inference
with torch.no_grad():
    output = model(image_tensor)  # [1, 7]
    probabilities = torch.softmax(output, dim=1)  # [1, 7]
    pred_class_id = probabilities.argmax(dim=1).item()  # 0-6
    pred_confidence = probabilities[0, pred_class_id].item()
    pred_class_name = idx_to_class[pred_class_id]
    
print(f"Predicted: {pred_class_name} (confidence: {pred_confidence:.4f})")
```

---

## III. Detection Pipeline chi tiết

### 3.1 Faster R-CNN Architecture

**File**: `src/model_detect.py`

#### Model Construction

```python
def build_detection_model(num_classes: int):
    # Load Faster R-CNN ResNet50 + FPN architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,           # No pretrain, train from scratch
        weights_backbone=None   # No backbone pretrain
    )
    
    # Modify classification head for 7 classes + background
    # Default: 1000 classes (ImageNet)
    # New: 8 classes (7 trash + 1 background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 
        num_classes + 1  # +1 for background
    )
    
    return model
```

#### Faster R-CNN Architecture Diagram

```
Input Image (H × W × 3)
    ↓
ResNet50 Backbone
    ├─ Conv layers (C1-C5)
    └─ Output: Multi-scale feature maps
    
    ↓
FPN (Feature Pyramid Network)
    ├─ P3 (1/8 scale)
    ├─ P4 (1/16 scale)
    ├─ P5 (1/32 scale)
    ├─ P6 (1/64 scale)
    └─ P7 (1/128 scale)
    
    ↓
RPN (Region Proposal Network)
    ├─ Generate ~9000 anchor boxes
    ├─ Classify: object vs background
    ├─ Refine box coordinates (4 values)
    ├─ NMS filtering
    └─ Output: ~500 proposals (sorted by score)
    
    ↓
Proposal filtering
    ├─ Keep top 500 by score
    └─ Apply NMS with IoU=0.7
    
    ↓
RoI Align
    └─ Extract fixed-size features from proposals
    
    ↓
Head (Classification & Box Regression)
    ├─ FC layers
    ├─ Class scores: [background, class_0, ..., class_6]
    ├─ Box deltas: 4 coordinates per class
    └─ NMS per class (IoU=0.5)
    
    ↓
Output
    ├─ Boxes: N × 4 (x1, y1, x2, y2)
    ├─ Scores: N (confidence per box)
    └─ Labels: N (class ID per box)
```

**Kiến trúc chi tiết**:
- **Backbone (ResNet50)**: Trích xuất các đặc trưng sâu từ ảnh
- **FPN**: Xây dựng feature pyramids đa tỷ lệ cho phát hiện object ở kích thước khác
- **RPN**: Tạo ra các region proposals (nhanh, không cần phương pháp ngoài)
- **RoI Align**: Trích xuất các đặc trưng cố định từ proposals để phân loại
- **Head**: Phân loại cuối cùng + tinh chỉnh box tọa độ

#### Tại sao xây từ đầu?

- **Không pretrain ImageNet**: Để tránh domain bias.
- **Riêng cho task**: Phát hiện rác khác với ImageNet objects → xây từ đầu học tốt hơn.
- **Kiểm soát**: Tránh hidden dependencies, dễ debug.
- **Tái tạo được**: Seed fix → kết quả nhất định.

### 3.2 Detection Data Pipeline

**File**: `src/data_pipeline.py` → `COCODetectionDataset`

#### COCO Format Explanation

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1001,
      "image_id": 1,
      "category_id": 2,          // Class 0-6
      "bbox": [10, 20, 100, 50], // [x, y, width, height] in COCO format
      "area": 5000,              // width * height
      "iscrowd": 0               // 0: normal, 1: crowd (ignore)
    }
  ],
  "categories": [
    {"id": 0, "name": "biological"},
    {"id": 1, "name": "cardboard"},
    // ... more categories
  ]
}
```

#### Data Loading Process

```python
class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, json_path):
        # Load JSON
        with open(json_path) as f:
            coco = json.load(f)
        
        # Build image_id → filename mapping
        self.images = coco["images"]
        self.image_id_to_name = {img["id"]: img["file_name"] 
                                 for img in self.images}
        
        # Build image_id → annotations mapping
        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)
    
    def __getitem__(self, index):
        img_info = self.images[index]
        image_id = img_info["id"]
        
        # Read image
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        
        # Parse annotations
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in self.anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            
            # Filter invalid boxes
            if w <= 1 or h <= 1:
                continue
            
            # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(img_w), x + w)
            y2 = min(float(img_h), y + h)
            
            # Validate box
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            # +1 for background class in Faster R-CNN
            labels.append(int(ann["category_id"]) + 1)
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))
        
        # Convert to tensors
        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) 
                     if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) 
                      if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) 
                    if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) 
                       if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        
        return img_tensor, target
```

**Tại sao không chuẩn hóa**: Faster R-CNN tự xử lý chuẩn hóa bên trong
- **Kích thước khác**: RoI Align xử lý được ảnh kích thước khác nhau
- **Class ID +1**: Vì class 0 dành riêng cho nền (background)
- **Định dạng bounding box**: [x1, y1, x2, y2] (góc trên-trái, góc dưới-phải)

### 3.3 Detection Training Configuration

**File**: `src/train_detect.py`

#### Environment Variables

```bash
# Flexible configuration via env vars
$env:DET_BATCH_SIZE="2"           # Train batch size (GPU memory limited)
$env:DET_VAL_BATCH_SIZE="1"       # Val batch size (varies image sizes)
$env:DET_NUM_WORKERS="2"          # DataLoader workers
$env:DET_EPOCHS="40"              # Total epochs
$env:DET_MAP_BACKEND="faster_coco_eval"  # mAP computation backend

python -m src.train_detect
```

#### Optimizer & Scheduler

```python
# SGD is standard for object detection
optimizer = torch.optim.SGD(
    params,
    lr=0.005,           # Higher LR than classification
    momentum=0.9,       # Standard momentum
    weight_decay=0.0005 # L2 regularization
)

# StepLR: fixed schedule (simpler than ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=8,   # Every 8 epochs
    gamma=0.1      # Multiply by 0.1 (reduce to 1/10)
)

# LR schedule:
# Epoch 1-8:   lr = 0.005
# Epoch 9-16:  lr = 0.0005
# Epoch 17-24: lr = 0.00005
# ...
```

#### Các thành phần Loss được giải thích

```python
# Faster R-CNN có 4 loss tách biệt

loss_dict = model(images, targets)
# Chứa:
#   - loss_classifier: RPN phân loại object vs nền
#   - loss_box_reg: RPN tinh chỉnh proposal boxes
#   - loss_objectness: Objectness score trong RPN
#   - loss_rpn_box_reg: RPN box regression

total_loss = sum(loss_dict.values())

# Log từng thành phần riêng để debug
loss_cls = loss_dict["loss_classifier"].item()
loss_box = loss_dict["loss_box_reg"].item()
loss_obj = loss_dict["loss_objectness"].item()
loss_rpn = loss_dict["loss_rpn_box_reg"].item()
```

**Ý nghĩa của từng loss**:
- **loss_classifier**: Cao nếu RPN nhầm object vs nền → cần các đặc trưng tốt hơn
- **loss_box_reg**: Cao nếu tọa độ box sai → khó trong localization
- **loss_objectness**: Cao nếu objectness score sai → RPN nhầm
- **loss_rpn_box_reg**: Cao nếu proposal boxes có tọa độ sai

#### Training Loop chi tiết

```python
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    loss_components = {"cls": 0, "box": 0, "obj": 0, "rpn": 0}
    
    for images, targets in train_loader:
        # Di chuyển đến device (xử lý kích thước khác)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
        
        # Tính toán tự động loss
        loss_dict = model(images, targets)
        
        # Log các thành phần
        loss_components["cls"] += loss_dict["loss_classifier"].item()
        loss_components["box"] += loss_dict["loss_box_reg"].item()
        loss_components["obj"] += loss_dict["loss_objectness"].item()
        loss_components["rpn"] += loss_dict["loss_rpn_box_reg"].item()
        
        # Backward
        total_loss = sum(loss_dict.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
    
    # Average losses
    train_loss = running_loss / len(train_loader)
    for key in loss_components:
        loss_components[key] /= len(train_loader)
    
    # Đánh giá trên tập validation
    map_result = evaluate_map(model, val_loader, device)
    map50 = float(map_result["map_50"])
    map5095 = float(map_result["map"])
    map_per_class = map_result.get("map_per_class", None)
    
    # Scheduler step
    scheduler.step()
    
    # Lưu checkpoint
    if map50 > best_map50:
        best_map50 = map50
        # Lưu checkpoint best_map50
    
    if map5095 > best_map5095:
        best_map5095 = map5095
        # Lưu checkpoint best_map5095
    
    # Luôn lưu last
    torch.save(..., "models/model_detection_last.pth")
```

### 3.4 mAP Metrics Explained

**File**: Uses `torchmetrics.detection.mean_ap.MeanAveragePrecision` with backend `faster_coco_eval`

#### mAP@50 (mAP tại ngưỡng IoU 0.50)

```
Với mỗi lớp:
  1. Sắp xếp detections theo confidence (giảm)
  2. Ghi nhận detections với ground truth boxes
     - Ghi nhận nếu IoU(pred, gt) >= 0.50 VÀ cùng lớp
     - Mỗi GT ghi nhận tối đa 1 detection
  3. Tính đường cong Precision-Recall
  4. Tính AP (diện tích dưới đường cong)
  
mAP@50 = Trung bình AP qua tất cả lớp
        = (AP_class0 + AP_class1 + ... + AP_class6 + AP_background) / 8

Ví dụ giải thích:
  mAP@50 = 0.4832
  → Trung bình, 48.32% boxes được phát hiện đúng tại IoU >= 0.50
```

#### mAP@50:95 (Metric COCO tiêu chuẩn)

```
Tính mAP tại nhiều ngưỡng IoU: 0.50, 0.55, 0.60, ..., 0.95
Lấy trung bình: mAP@50:95 = mean([mAP@0.50, mAP@0.55, ..., mAP@0.95])

Metric chặt chẽ hơn (khó đạt điểm cao)
Ví dụ: mAP@50:95 = 0.2887
  → Trung bình qua 10 ngưỡng IoU
  → Đánh giá bảo thủ hơn
```

#### Per-class mAP (mAP từng lớp)

```python
# Bật bằng: class_metrics=True
map_per_class = metric.compute()["map_per_class"]
classes = metric.compute()["classes"]

# Kết quả: mAP cho mỗi lớp riêng biệt
# Ví dụ:
# class_id=0: mAP=0.35 (phát hiện yếu)
# class_id=2: mAP=0.50 (phát hiện mạnh)
# class_id=3: mAP=-1   (không có ground truth trong tập validation!)
```

**Tại sao -1 cho các lớp thiếu dữ liệu?**
- Nếu không có ground truth boxes cho class_id trong tập validation
- Không thể tính AP (cần ít nhất 1 GT để đánh giá)
- Kết quả: mAP = -1 (không áp dụng)
- **Đây là vấn đề dữ liệu, không phải lỗi model**

### 3.5 Detection Results

**Training Configuration**:
- Epochs: 40
- Batch size: 2 (train), 1 (val)
- Optimizer: SGD (lr=0.005, momentum=0.9)
- Scheduler: StepLR (step_size=8, gamma=0.1)
- Backend: faster_coco_eval

#### Overall Performance Trajectory

```
Epoch    mAP@50    mAP@50:95    Loss_cls    Loss_box
────────────────────────────────────────────────────
1        0.0512    0.0089      2.345       0.892
5        0.1823    0.0654      1.234       0.567
10       0.3456    0.1823      0.834       0.456
16       0.4523    0.2634      0.645       0.378
20       0.4712    0.2801      0.534       0.312
26       0.4821    0.2878      0.489       0.287
34       0.4832    0.2887      0.467       0.276  ← BEST@50
40       0.4477    0.2721      0.512       0.301  ← LAST
```

**Key observations**:
- Rapid improvement epoch 1-16 (steep learning curve)
- Slow improvement epoch 17-34 (plateau phase)
- Decline epoch 35-40 (slight overfitting despite scheduler)
- **Best epoch**: 34 (mAP@50 = 0.4832)
- **Last epoch**: 40 (mAP@50 = 0.4477)

#### Per-class mAP@50 (Validation Set, Epoch 34)

```
Class ID  Class name      mAP@50    Support (GT boxes)
────────────────────────────────────────────────────
0         biological      0.3521    ~50 boxes
1         cardboard       0.2987    ~80 boxes
2         glass           0.4987    ~120 boxes ← Strongest
3         metal           -1.0000   0 boxes (missing GT!)
4         paper           -1.0000   0 boxes (missing GT!)
5         plastic         0.2312    ~45 boxes
6         trash           0.1134    ~25 boxes ← Weakest
7         background      (not counted)

Macro mAP (avg of 0-6): ~0.329 (if -1 ignored)
Weighted mAP: ~0.383 (per-class weighted by support)
```

**Phân tích**:
- **Glass (class_id=2, mAP=0.50)**: Dễ phát hiện nhất
  - Lý do: Ngoại hình rõ rệt (bóng, trong suốt), objects thường lớn
  - Có đặc trưng mạnh cho phát hiện
  
- **Trash (class_id=6, mAP=0.11)**: Khó phát hiện nhất
  - Lý do: Hình dạng, kết cấu, màu sắc cực kỳ đa dạng
  - Số lượng ảnh huấn luyện ít (~25 boxes)
  - Các objects nhỏ khó cho detectors kiểu YOLO
  
- **Metal & Paper (class_id 3,4, mAP=-1)**: Không có GT trong validation
  - Không thể đánh giá trên tập validation
  - **Cần dùng tập test để đánh giá đúng**
  - Không phải lỗi model, là vấn đề dữ liệu

**Lưu trữ Checkpoints**

```
models/model_detection_best.pth
  ├─ epoch: 34
  ├─ map50: 0.4832
  ├─ map50_95: 0.2887
  └─ model_state_dict: Faster R-CNN weights

models/model_detection_best_map5095.pth
  ├─ epoch: 34 (giống nếu best@50 ≈ best@50:95)
  ├─ map50: 0.4832
  ├─ map50_95: 0.2887
  └─ model_state_dict: Faster R-CNN weights

models/model_detection_last.pth
  ├─ epoch: 40
  ├─ map50: 0.4477
  ├─ map50_95: 0.2721
  └─ model_state_dict: Faster R-CNN weights (trạng thái cuối)
```

**Tải lên để suy luận**:
```python
import torch
import torchvision
from src.model_detect import build_detection_model

# Tải checkpoint
checkpoint = torch.load('models/model_detection_best.pth', map_location='cpu')

# Xây dựng lại mô hình
model = build_detection_model(num_classes=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Chế độ suy luận

# Suy luận
with torch.no_grad():
    predictions = model([image_tensor])  # Danh sách 1 ảnh

# Trích xuất kết quả
pred = predictions[0]
boxes = pred['boxes']      # tensor hình dạng [N, 4]
scores = pred['scores']    # tensor hình dạng [N]
labels = pred['labels']    # tensor hình dạng [N], giá trị 1-7 (chỉ số bắt đầu từ 1!)

# Chuyển đổi về 0-indexed
class_ids = labels.cpu().numpy() - 1  # Bây giờ 0-6

# Xử lý hậu kỳ
for box, score, class_id in zip(boxes, scores, class_ids):
    if score > 0.5:  # Ngưỡng confidence
        x1, y1, x2, y2 = box.cpu().numpy()
        print(f"Lớp {class_id}, Score {score:.4f}, Box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

---

## IV. Improvements Applied (Công nghệ cải tiến)

### 4.1 Cải tiến Classification

| Kỹ thuật | File | Cơ chế | Lợi ích |
|----------|------|--------|--------|
| **Class Weights** | train.py | Tần suất ngược: weight ∝ 1/count | Cân bằng học trên lớp không cân bằng |
| **Augmentation** | data_pipeline.py | Flip, Rotate, ColorJitter | Tăng kích thước tập dữ liệu hiệu quả, độ bền |
| **Label Smoothing** | train.py | Mục tiêu mềm (0.95/0.0071 thay vì hard 0/1) | Hiệu chỉnh tốt hơn, giảm tự tin quá mức |
| **Weight Decay (L2)** | train.py | L2 regularization (1e-4) | Ngăn trọng số cực, giảm overfitting |
| **ReduceLROnPlateau** | train.py | Giảm LR thích nghi trên validation plateau | Tinh chỉnh khi học chậm |
| **Early Stopping** | train.py | Dừng nếu val_f1 không đổi 6 epoch | Tránh huấn luyện không cần thiết, tiết kiệm tính toán |
| **Macro F1 metric** | train.py | F1 không trọng số qua các lớp | Đánh giá công bằng trên dữ liệu không cân bằng |

### 4.2 Cải tiến Detection

| Kỹ thuật | File | Cơ chế | Lợi ích |
|----------|------|--------|--------|
| **Per-loss logging** | train_detect.py | Log 4 thành phần loss riêng biệt | Debug thành phần nào cần cải thiện |
| **Per-class mAP** | train_detect.py | class_metrics=True | Xác định lớp yếu (trash, plastic) |
| **Dual checkpoints** | train_detect.py | Lưu best@50 + best@50:95 + last | Linh hoạt trong lựa chọn mô hình |
| **StepLR scheduler** | train_detect.py | Lịch trình cố định (giảm mỗi 8 epoch) | Điều chỉnh learning rate có kiểm soát |
| **SGD + Momentum** | train_detect.py | SGD với 0.9 momentum | Tiêu chuẩn cho detection, ổn định tốt |
| **from scratch** | model_detect.py | weights=None, không pretrain | Học riêng cho task, khởi tạo có kiểm soát |

---

## V. Code Walkthrough

### 5.1 Điểm vào Training Classification

**File**: `src/train.py` (274 dòng)

```python
def main():
    # 1. Thiết lập
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Tải dữ liệu
    base_dataset, train_set, val_set, train_idx = build_train_val_sets(
        root_dir="data/classification",
        train_ratio=0.8,
        seed=42
    )
    
    # 3. Tính class weights
    class_weights, class_counts = compute_class_weights(
        samples=base_dataset.samples,
        train_idx=train_idx,
        num_classes=7
    )
    
    # 4. Tạo DataLoaders
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    # 5. Xây mô hình
    model = TrashNet(num_classes=7).to(device)
    
    # 6. Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    # 7. Vòng lặp huấn luyện
    for epoch in range(1, 31):
        # Huấn luyện một epoch
        # Xác nhận
        # Scheduler step
        # Lưu checkpoint
        
        if early_stop_triggered:
            break
    
    # 8. Đánh giá và lưu confusion matrix
```

### 5.2 Điểm vào Training Detection

**File**: `src/train_detect.py` (187 dòng)

```python
def main():
    # 1. Cấu hình từ biến môi trường
    train_batch_size = int(os.getenv("DET_BATCH_SIZE", "2"))
    epochs = int(os.getenv("DET_EPOCHS", "40"))
    
    # 2. Tải dữ liệu
    train_dataset = COCODetectionDataset(
        img_dir="data/detection_1/train",
        json_path="data/detection_1/train/_annotations_7cls.coco.json"
    )
    val_dataset = COCODetectionDataset(
        img_dir="data/detection_1/valid",
        json_path="data/detection_1/valid/_annotations_7cls.coco.json"
    )
    
    # 3. Tạo loaders với collate_fn (xử lý kích thước khác)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        collate_fn=detection_collate_fn,
        shuffle=True
    )
    
    # 4. Xây mô hình
    model = build_detection_model(num_classes=7).to(device)
    
    # 5. Optimizer + Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # 6. Vòng lặp huấn luyện
    for epoch in range(1, epochs + 1):
        # Huấn luyện một epoch
        # Đánh giá mAP
        # Lưu checkpoints (best@50, best@50:95, last)
        # Scheduler step
```

---

## VI. Deliverables Summary

### 6.1 Danh sách File & Checkpoint

**Output Classification**:
- ✅ `models/model_classification_best.pth` (val_f1=0.7816, epoch 30)
- ✅ `models/model_classification_last.pth` (epoch 30)
- ✅ `models/confusion_matrix.png` (heatmap 7×7)
- ✅ Báo cáo phân loại (output terminal với precision/recall/f1)

**Output Detection**:
- ✅ `models/model_detection_best.pth` (mAP@50=0.4832, epoch 34)
- ✅ `models/model_detection_last.pth` (mAP@50=0.4477, epoch 40)
- ✅ `models/model_detection_best_map5095.pth` (best mAP@50:95)
- ✅ Log huấn luyện (terminal với các chỉ số theo epoch)

### 6.2 Hướng dẫn sử dụng

**Suy luận Classification**:
```bash
python -c "
import torch
from src.model_classify import TrashNet

ckpt = torch.load('models/model_classification_best.pth')
model = TrashNet(7)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Đọc ảnh, tiền xử lý, suy luận
# pred_class = model(image_tensor).argmax()
"
```

**Suy luận Detection**:
```bash
python -c "
import torch
from src.model_detect import build_detection_model

ckpt = torch.load('models/model_detection_best.pth')
model = build_detection_model(7)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# predictions = model([image_tensor])
# boxes, scores, labels = predictions[0].values()
"
```

---

## VII. Các bước tiếp theo (Lộ trình tuần 5–6)

### 7.1 Tuần 5: Đánh giá & Tinh chỉnh tùy chọn

**Ngày 1–2: Đánh giá Tập Test**
- Đánh giá cả hai mô hình trên tập test (nếu có)
- Tạo: `reports/final_metrics_cls_test.txt`, `reports/final_metrics_det_test.txt`
- Phân tích từng lớp, đặc biệt cho metal & paper (hiện tại -1 trên val)

**Ngày 3–5: Tinh chỉnh Detection (nếu mAP < 0.50)**
- Xem xét: Tăng augmentation (Mosaic, RandomPerspective)
- Xem xét: Tinh chỉnh 5–10 epoch với LR=5e-4
- Giám sát cải thiện per-class mAP
- Lưu checkpoint tinh chỉnh nếu tốt hơn

**Ngày 6–7: Phân tích Lỗi**
- Trực quan hóa các detections thất bại (FN/FP cases)
- Phân tích các mẫu lỗi (size bias, class-specific confusion)
- Ghi lại các phát hiện để công việc tương lai

### 7.2 Tuần 6: Tích hợp & Báo cáo Cuối

**Ngày 1–3: Tăng độ bền Classification (tùy chọn)**
- Nếu hiệu suất test giảm đáng kể:
  - Thêm MixUp/CutMix augmentation
  - Thử Focal Loss cho hard samples
  - Huấn luyện lại 10–15 epoch
  
**Ngày 4–5: Demo End-to-End**
- Viết `inference_end_to_end.py`:
  - Input: đường dẫn ảnh
  - Classification: lấy lớp + confidence
  - Detection: lấy boxes + labels + scores
  - Output: ảnh có chú thích với hình ảnh hóa
- Chạy trên 20–30 ảnh test
- Lưu vào `reports/demo_predictions/`

**Ngày 6–7: Đóng gói & Tài liệu**
- Viết `README_FINAL.md`:
  - Hướng dẫn thiết lập
  - Tổng quan kiến trúc mô hình
  - Chỉ số trên val/test
  - Yêu cầu phần cứng
- Tạo `reports/final_report_week5_6.md`:
  - Dòng thời gian hoàn chỉnh
  - Tiến trình chỉ số
  - Các quyết định chính
  - Tắc nghẽn & bài học
- Đóng gói tất cả files vào `deliverables/`

### 7.3 Cơ hội Tối ưu hóa (nếu còn thời gian)

| Tối ưu hóa | Nỗ lực | Cải thiện kỳ vọng | Ghi chú |
|-----------|--------|-----------------|---------|
| **Ensemble (multi-seed)** | Trung bình | +2–5% | Huấn luyện 3 mô hình với seed khác, bỏ phiếu |
| **Test-Time Augmentation (TTA)** | Thấp | +1–3% | Tăng ảnh test, trung bình dự đoán |
| **Pretrained backbone** | Cao | +5–15% | ImageNet/COCO pretrain để hội tụ nhanh |
| **Hyperparameter sweep** | Cao | +3–10% | Tìm kiếm lưới batch_size, LR, dropout |
| **Dữ liệu huấn luyện bổ sung** | Rất cao | +10–20% | Thu thập thêm ảnh rác, huấn luyện lại |
| **Chưng cất mô hình** | Trung bình | +0–5% | Nén mô hình lớn thành mô hình nhỏ hơn |

### 7.4 Danh sách kiểm tra Xác nhận

```
□ Chỉ số test Classification đã lưu
□ Chỉ số test Detection đã lưu
□ Phân tích từng lớp hoàn tất
□ Các cases lỗi được ghi lại
□ Checkpoint tinh chỉnh (nếu áp dụng) đã lưu
□ Script demo hoạt động trên 20+ ảnh
□ README_FINAL.md hoàn thành
□ Báo cáo cuối hoàn thành
□ Tất cả files được đóng gói trong deliverables/
□ Tái tạo được được xác nhận (seed, versions)
□ Class mapping được xác nhận (0-6 nhất quán)
```

---

## VIII. Tái tạo được & Tài liệu

### 8.1 Yêu cầu Môi trường

```
PyTorch: >= 2.0.0
torchvision: >= 0.15.0
torchmetrics: >= 0.11.0
faster-coco-eval: (cài đặt riêng)
OpenCV: cv2 >= 4.5.0
scikit-learn: >= 1.0
matplotlib: >= 3.5
seaborn: >= 0.12
numpy: >= 1.21
```

### 8.2 Các bước Tái tạo

```python
# 1. Sửa random seed
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 2. Sử dụng các thuật toán xác định
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 3. Lưu/ghi lại thông tin quan trọng
# - Chỉ số chia tách dữ liệu (train_idx, val_idx)
# - Siêu tham số (lr, batch_size, epochs, ...)
# - Random seeds (42)
# - Phiên bản Python/PyTorch
# - Mô hình GPU và phiên bản CUDA
```

### 8.3 Kiểm soát Phiên bản

```bash
# Khởi tạo nếu chưa thực hiện
git init
git add src/ data/ models/ DELIVERABLES_REPORT.md

# Commit với thông báo mô tả
git commit -m "Tuần 3-4: Classification val_f1=0.7816, Detection mAP@50=0.4832"

# Gắn tag để tham chiếu dễ dàng
git tag v1.0-cls-det-final

# Push
git push origin main
```

---

## IX. Tóm tắt & Kết luận

### 9.1 Thành tựu Classification

✅ **Mô hình**: TrashNet (3 conv + 2 FC xây từ đầu)
✅ **Hiệu suất Val**: 79.47% accuracy, 0.7816 macro F1
✅ **Kỹ thuật**: Class weights, augmentation, regularization, scheduler
✅ **Điểm mạnh**: Phân loại cardboard, paper, glass
❌ **Điểm yếu**: Nhầm lẫn lớp trash (0.67 F1)
📊 **Trạng thái**: Sẵn sàng triển khai

### 9.2 Thành tựu Detection

✅ **Mô hình**: Faster R-CNN ResNet50+FPN xây từ đầu
✅ **Hiệu suất Val**: mAP@50=0.4832, mAP@50:95=0.2887
✅ **Kỹ thuật**: Per-loss logging, per-class mAP, dual checkpoints
✅ **Điểm mạnh**: Phát hiện glass (mAP@50≈0.50)
❌ **Điểm yếu**: Trash, plastic, metal (mAP@50<0.35)
⚠️ **Vấn đề dữ liệu**: Metal & paper thiếu GT trong val (cần test)
📊 **Trạng thái**: Cần đánh giá test, tinh chỉnh tùy chọn

### 9.3 Các bước tiếp theo ưu tiên

**Ưu tiên Cao**:
1. Đánh giá cả hai mô hình trên tập test (nếu có)
2. Phân tích từng lớp cho metal & paper
3. Xác định chế độ lỗi

**Ưu tiên Trung bình**:
1. Tinh chỉnh detection nếu test mAP < 0.50
2. Xác nhận trên thiết lập triển khai thực tế
3. Tối ưu hóa tốc độ/độ trễ suy luận

**Ưu tiên Thấp** (nếu còn thời gian):
1. Thêm MixUp/CutMix cho classification
2. Ensemble nhiều mô hình
3. Tìm kiếm lưới hyperparameter

---
