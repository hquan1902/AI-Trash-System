import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

# Try importing plotly
try:
    # pyrefly: ignore [missing-import]
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

from src.detection.models.model_detect import build_detection_model, apply_nms_to_prediction

# Constants
NUM_CLASSES = 7

# User's ResNet50 Classes
USER_CLASS_NAMES = ["cardboard", "glass", "metal", "organic", "paper", "plastic", "trash"]

# Friend's Faster R-CNN Classes (original names mapping to 1-7 output labels)
DET_CLASS_NAMES = ["biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"]

# User's visualization and Vietnamese mapping config (taken from test_model.py)
CLASS_INFO = {
    "cardboard": {"vi": "Bìa / Carton",      "icon": "📦", "color": "#D4A96A"},
    "glass":     {"vi": "Thủy tinh",          "icon": "🫙", "color": "#68C9C9"},
    "metal":     {"vi": "Kim loại",           "icon": "🥫", "color": "#9BB5C8"},
    "organic":   {"vi": "Hữu cơ",            "icon": "🍃", "color": "#6DBF6D"},
    "paper":     {"vi": "Giấy",              "icon": "📄", "color": "#E2D080"},
    "plastic":   {"vi": "Nhựa",              "icon": "♻️", "color": "#78B8E8"},
    "trash":     {"vi": "Rác thông thường",  "icon": "🗑️", "color": "#B09898"},
}

RECYCLE_TIP = {
    "cardboard": "Gấp dẹp, giữ khô trước khi bỏ vào thùng tái chế.",
    "glass":     "Rửa sạch, không bỏ lẫn với nhựa hay kim loại.",
    "metal":     "Bóp dẹp lon/hộp để tiết kiệm không gian.",
    "organic":   "Có thể ủ phân compost — tốt cho đất trồng cây.",
    "paper":     "Tránh để ướt, xếp gọn trước khi mang đi tái chế.",
    "plastic":   "Kiểm tra ký hiệu tái chế ở đáy trước khi bỏ.",
    "trash":     "Bọc kín, bỏ vào thùng rác thông thường.",
}

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align_class_name(class_name: str) -> str:
    """Map class names to unify them with the user's classes (e.g. 'biological' -> 'organic')."""
    if class_name == "biological":
        return "organic"
    return class_name

# MODEL LOADERS

def load_user_classifier(model_path: str = "models/best_resnet50.pth") -> torch.nn.Module:
    """Load the user's ResNet50 classification model."""
    device = get_device()
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model classification tại: {model_path}")
        
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m

def load_friend_detector(model_path: str = "models/model_detection_pretrained_best_map50.pth") -> torch.nn.Module:
    """Load the friend's Faster R-CNN detection model."""
    device = get_device()
    
    if not os.path.exists(model_path):
        basename = os.path.basename(model_path)
        fallback_path = os.path.join("models", basename)
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy model detection tại: {model_path} hoặc {fallback_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # Parse checkpoint details if dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        variant = checkpoint.get("model_variant", None)
        if variant is None:
            if "best_7_4" in os.path.basename(model_path):
                variant = "v1"
            else:
                variant = "v2"
        else:
            variant = str(variant).lower().strip()
            
        use_pretrained = checkpoint.get("use_pretrained", None)
        if use_pretrained is None:
            if "best_7_4" in os.path.basename(model_path):
                use_pretrained = False
            else:
                use_pretrained = True
        else:
            use_pretrained = bool(use_pretrained)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        if "best_7_4" in os.path.basename(model_path):
            variant = "v1"
            use_pretrained = False
        else:
            variant = "v2"
            use_pretrained = True
    else:
        raise ValueError(f"Định dạng checkpoint không được hỗ trợ: {type(checkpoint)}")
        
    # Build model using the parameters extracted
    model = build_detection_model(
        num_classes=NUM_CLASSES,
        use_pretrained=use_pretrained,
        variant=variant
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Keep thresholds low for flexibility (we will filter dynamically using our sliders)
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh = 0.7
    
    return model

# INFERENCE PIPELINES

def predict_classification(img: Image.Image, model: torch.nn.Module) -> Tuple[np.ndarray, List[int]]:
    """Run full image classification using the ResNet50 model."""
    device = get_device()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
    top3 = probs.argsort()[::-1][:3].tolist()
    return probs, top3

@torch.no_grad()
def run_detection(
    model: torch.nn.Module,
    pil_image: Image.Image,
    score_threshold: float = 0.1,
    nms_iou_threshold: float = 0.4,
) -> Dict[str, torch.Tensor]:
    """Run object detection on the input image using Faster R-CNN."""
    device = get_device()
    
    # Normalize image to [C, H, W] tensor in [0, 1]
    rgb_image = pil_image.convert("RGB")
    image_np = np.array(rgb_image)
    image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
    
    outputs = model([image_tensor.to(device)])
    prediction = {k: v.detach().cpu() for k, v in outputs[0].items()}
    
    # Apply custom NMS post-processing
    prediction = apply_nms_to_prediction(
        prediction,
        iou_threshold=float(nms_iou_threshold),
        score_threshold=float(score_threshold),
    )
    return prediction

@torch.no_grad()
def classify_crops(
    image_np: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> List[Dict[str, Any]]:
    """Crop each bounding box and classify it using the ResNet50 classification model."""
    device = get_device()
    results: List[Dict[str, Any]] = []
    boxes = prediction.get("boxes", torch.empty((0, 4)))
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(image_np.shape[1] - 1, x1))
        y1 = max(0, min(image_np.shape[0] - 1, y1))
        x2 = max(1, min(image_np.shape[1], x2))
        y2 = max(1, min(image_np.shape[0], y2))
        
        if x2 <= x1 or y2 <= y1:
            results.append({"class": "trash", "score": 0.0, "probs": np.zeros(NUM_CLASSES)})
            continue
            
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            results.append({"class": "trash", "score": 0.0, "probs": np.zeros(NUM_CLASSES)})
            continue
            
        crop_pil = Image.fromarray(crop).convert("RGB")
        t = tfm(crop_pil).unsqueeze(0).to(device)
        
        logits = model(t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        
        results.append({
            "class": USER_CLASS_NAMES[pred_idx],
            "score": float(probs[pred_idx]),
            "probs": probs
        })
        
    return results


# FUSION & MAPPING LOGIC

def fuse_detection_and_classification(
    det_class: str,
    det_score: float,
    cls_class: str,
    cls_score: float,
    cls_override_threshold: float = 0.65,
) -> Tuple[str, float, str]:
    """
    Fuse Faster R-CNN detection class and ResNet50 classification crop class.
    Both inputs are standard class names (with biological aligned to organic).
    """
    det_aligned = align_class_name(det_class)
    cls_aligned = align_class_name(cls_class)
    
    # Neu ca hai mo hinh dong y voi nhau ve loai rac
    if det_aligned == cls_aligned:
        # Lay diem tin cay cao hon trong ca hai
        final_score = max(det_score, cls_score)
        return det_aligned, final_score, "agree"
        
    # Neu ResNet50 doan khac nhung lai rat tu tin (vuot nguong ghi de)
    if cls_score >= cls_override_threshold:
        # Thay doi sang loai rac do ResNet50 phan loai
        return cls_aligned, cls_score, "cls_override"
        
    # Mac dinh neu ResNet50 doan khac nhung khong tin cay bang, giu nguyen Faster R-CNN
    return det_aligned, det_score, "det_keep"

# VISUALIZATION UTILITIES

def draw_static_predictions(
    image_np: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    class_names: List[str] = DET_CLASS_NAMES,
) -> np.ndarray:
    """Draw bounding boxes using standard PIL library to get a static preview image."""
    canvas = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(canvas)
    
    boxes = prediction.get("boxes", torch.empty((0, 4)))
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.int64))
    scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        
        # detector labels are 1-based (0 is background)
        class_idx = int(label.item()) - 1
        raw_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"cls_{int(label.item())}"
        aligned_name = align_class_name(raw_name)
        
        # Display Info
        info = CLASS_INFO.get(aligned_name, {"vi": aligned_name, "color": "#00FF00", "icon": "🗑️"})
        color_hex = info["color"].lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        
        text = f"{info['icon']} {info['vi']}: {float(score.item()):.2f}"
        
        # Draw box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=3)
        
        # Draw background for text to make it readable
        try:
            # Try to get font size
            text_size = draw.textbbox((x1, y1), text)
            tw = text_size[2] - text_size[0]
            th = text_size[3] - text_size[1]
        except AttributeError:
            # Fallback for older PIL versions
            tw, th = draw.textsize(text) if hasattr(draw, "textsize") else (80, 12)
            
        draw.rectangle([(x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1)], fill=color_rgb)
        draw.text((x1 + 4, max(0, y1 - th - 3)), text, fill=(255, 255, 255))
        
    return np.array(canvas)

def draw_fused_predictions(
    image_np: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    cls_results: List[Dict[str, Any]],
    cls_override_threshold: float,
) -> np.ndarray:
    """Draw bounding boxes showing the fused (detector + classifier) labels."""
    canvas = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(canvas)
    
    boxes = prediction.get("boxes", torch.empty((0, 4)))
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.int64))
    scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32))
    
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        
        # Detector prediction details
        det_idx = int(label.item()) - 1
        det_name = DET_CLASS_NAMES[det_idx] if 0 <= det_idx < len(DET_CLASS_NAMES) else f"cls_{int(label.item())}"
        det_score = float(score.item())
        
        # Fused details
        if idx < len(cls_results):
            cls_name = cls_results[idx]["class"]
            cls_score = cls_results[idx]["score"]
            final_class, final_score, fusion_source = fuse_detection_and_classification(
                det_name, det_score, cls_name, cls_score, cls_override_threshold
            )
        else:
            final_class, final_score, fusion_source = align_class_name(det_name), det_score, "det_only"
            
        info = CLASS_INFO.get(final_class, {"vi": final_class, "color": "#00FF00", "icon": "🗑️"})
        color_hex = info["color"].lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        
        # Show fusion tag
        tag = ""
        if fusion_source == "cls_override":
            tag = " ✏️"
        elif fusion_source == "agree":
            tag = " 🤝"
            
        text = f"{info['icon']} {info['vi']}: {final_score:.2f}{tag}"
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=3)
        
        try:
            text_size = draw.textbbox((x1, y1), text)
            tw = text_size[2] - text_size[0]
            th = text_size[3] - text_size[1]
        except AttributeError:
            tw, th = draw.textsize(text) if hasattr(draw, "textsize") else (80, 12)
            
        draw.rectangle([(x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1)], fill=color_rgb)
        draw.text((x1 + 4, max(0, y1 - th - 3)), text, fill=(255, 255, 255))
        
    return np.array(canvas)

def build_interactive_bbox_figure(
    image_np: np.ndarray,
    rows: List[Dict[str, Any]],
):
    """Generate a Plotly-based interactive visualization allowing box hover inspection."""
    if not PLOTLY_AVAILABLE:
        return None
        
    height, width = image_np.shape[:2]
    fig = go.Figure()
    
    # Add background image
    fig.add_layout_image(
        dict(
            source=Image.fromarray(image_np),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below",
        )
    )
    
    for row in rows:
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        hover_text = row.get("hover_text", "")
        color = row.get("color", "lime")
        
        fig.add_trace(
            go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[y1, y1, y2, y2, y1],
                mode="lines",
                line=dict(color=color, width=2.5),
                fill=None,
                hovertemplate=f"{hover_text}<extra></extra>",
                showlegend=False,
            )
        )
        
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[height, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=min(900, max(360, int(height * 0.6))),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
