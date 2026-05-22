# AI-Trash-System
AI system for trash detection and classification using CNN, OCR, and Grad-CAM.

## Overview
This project separates **detection** and **classification** training into clear modules.
You can retrain using the new structure and still get the same type of JSONs and
model checkpoints as before.

## Project Structure (organized)
```
AI_TRASH_SYSTEM/
├── data/
│   ├── raw/                    # data gốc chưa xử lý
│   ├── processed/              # data đã remap/clean
│   ├── detection/              # detection dataset hiện tại (COCO)
│   └── classification/         # classification dataset hiện tại
├── models/
│   ├── pretrained/             # pretrained detection checkpoints
│   ├── detection/              # curated detection checkpoints (optional)
│   └── classification/         # curated classification checkpoints (optional)
├── src/
│   ├── data/                   # data pipeline scripts
│   │   ├── build_dataset.py    # merge detection datasets → COCO
│   │   ├── remap_labels.py     # remap labels to 7 classes
│   │   └── split_dataset.py    # train/val split
│   ├── detection/              # detection training
│   │   └── train.py
│   ├── classification/         # classification training
│   │   └── train.py
│   └── train_detect.py          # detection entrypoint (same logic)
│   └── train.py                 # classification entrypoint (same logic)
├── web/                        # Streamlit app
├── notebooks/
├── reports/
├── requirements.txt
└── README.md
```

## Detection Pipeline
```powershell
# 1) Merge datasets (COCO)
python -m src.data.build_dataset

# 2) Remap labels to 7 classes
python -m src.data.remap_labels

# 3) Split train/val
python -m src.data.split_dataset

# 4) Train detection (choose one)
python -m src.detection.train
python -m src.train_detect
```

## Classification Pipeline
```powershell
# Train classification
python -m src.classification.train
python -m src.train
```

## System Architecture
Input Image
     │
     ▼
Detection Model (CNN)
     │
     ▼
Bounding Boxes
     │
     ▼
OpenCV Crop
     │
     ▼
Classification Model (CNN)
     │
     ▼
Fusion Logic
     │
     ▼
Final Prediction
     │
     ▼
Streamlit Web Interface
     ▼
Streamlit Web Interface