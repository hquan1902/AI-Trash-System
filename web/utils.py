import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
	import plotly.graph_objects as go
	PLOTLY_AVAILABLE = True
except Exception:
	go = None
	PLOTLY_AVAILABLE = False

from src.model_classify import TrashNet
from src.model_detect import build_detection_model, apply_nms_to_prediction


# Mặc định theo dataset classification/detection hiện tại của dự án.
DEFAULT_CLASS_NAMES = [
	"biological",
	"cardboard",
	"glass",
	"metal",
	"paper",
	"plastic",
	"trash",
]


def get_device() -> str:
	return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_path(model_path: str | None = None) -> str:
	if model_path and os.path.exists(model_path):
		return model_path

	candidates = [
		"models/model_detection_best_7_4.pth",
		"models/model_detection_best_map5095_7_4.pth",
		"models/pretrained/model_detection_pretrained_best_map5095.pth",
		"models/pretrained/model_detection_pretrained_best_map50.pth",
		"models/pretrained/model_detection_pretrained_last.pth",
		"models/model_detection_last.pth",
	]
	for path in candidates:
		if os.path.exists(path):
			return path

	raise FileNotFoundError(
		"Không tìm thấy checkpoint detection. Hãy đảm bảo có một trong các file: "
		"models/model_detection_best_7_4.pth, models/model_detection_best_map5095_7_4.pth, "
		"models/pretrained/model_detection_pretrained_best_map5095.pth, "
		"models/pretrained/model_detection_pretrained_best_map50.pth hoặc models/model_detection_last.pth"
	)


def _resolve_classification_model_path(model_path: str | None = None) -> str:
	if model_path and os.path.exists(model_path):
		return model_path

	candidates = [
		"models/model_classification_best.pth",
		"models/model_classification_last.pth",
		"models/model_classification.pth",
	]
	for path in candidates:
		if os.path.exists(path):
			return path

	raise FileNotFoundError(
		"Không tìm thấy checkpoint classification. Hãy đảm bảo có một trong các file: "
		"models/model_classification_best.pth hoặc model_classification_last.pth hoặc model_classification.pth"
	)


def load_detection_model(
	model_path: str | None = None,
	num_classes: int = 7,
) -> Tuple[torch.nn.Module, str, str]:
	"""Load Faster R-CNN checkpoint để infer 1 ảnh."""
	resolved_model_path = _resolve_model_path(model_path)
	device = get_device()
	checkpoint = torch.load(resolved_model_path, map_location=device)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
		ckpt_variant = str(checkpoint.get("model_variant", "v2")).lower().strip() or "v2"
		ckpt_use_pretrained = bool(checkpoint.get("use_pretrained", False))
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
		ckpt_variant = "v2"
		ckpt_use_pretrained = False
	else:
		raise ValueError(f"Checkpoint format không hỗ trợ: {type(checkpoint).__name__}")

	candidate_builds = [
		(ckpt_variant, ckpt_use_pretrained),
		("v2", ckpt_use_pretrained),
		("v2", False),
		("v1", False),
		("v1", True),
	]

	seen = set()
	ordered_candidates = []
	for cfg in candidate_builds:
		if cfg not in seen:
			seen.add(cfg)
			ordered_candidates.append(cfg)

	last_error: Exception | None = None
	model = None
	for variant, use_pretrained in ordered_candidates:
		try:
			model = build_detection_model(
				num_classes=num_classes,
				use_pretrained=use_pretrained,
				variant=variant,
			)
			model.load_state_dict(state_dict)
			break
		except Exception as exc:
			last_error = exc
			model = None

	if model is None:
		raise RuntimeError(
			f"Không thể load checkpoint detection '{resolved_model_path}'. "
			f"Đã thử nhiều cấu hình model nhưng đều thất bại. Lỗi cuối: {last_error}"
		)

	model.to(device)
	model.eval()

	# Giữ ngưỡng nội bộ permissive để slider ở app điều khiển hậu xử lý linh hoạt.
	model.roi_heads.score_thresh = 0.001
	model.roi_heads.nms_thresh = 0.7

	return model, device, resolved_model_path


def load_classification_model(
	model_path: str | None = None,
) -> Tuple[torch.nn.Module, str, str, Dict[int, str]]:
	"""Load classification checkpoint để vote lại class cho từng bbox."""
	resolved_model_path = _resolve_classification_model_path(model_path)
	device = get_device()

	checkpoint = torch.load(resolved_model_path, map_location=device)
	num_classes = int(checkpoint.get("num_classes", len(DEFAULT_CLASS_NAMES)))

	model = TrashNet(num_classes=num_classes)
	if "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"])
	else:
		model.load_state_dict(checkpoint)

	model.to(device)
	model.eval()

	if "class_to_idx" in checkpoint:
		idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
	else:
		idx_to_class = {i: name for i, name in enumerate(DEFAULT_CLASS_NAMES[:num_classes])}

	return model, device, resolved_model_path, idx_to_class


def prepare_image(pil_image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
	"""Chuẩn hóa ảnh đầu vào về tensor [C, H, W] float32 trong [0,1]."""
	rgb_image = pil_image.convert("RGB")
	image_np = np.array(rgb_image)
	image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
	return image_np, image_tensor


@torch.no_grad()
def run_detection(
	model: torch.nn.Module,
	image_tensor: torch.Tensor,
	device: str,
	score_threshold: float = 0.1,
	nms_iou_threshold: float = 0.4,
) -> Dict[str, torch.Tensor]:
	"""Chạy suy luận và áp dụng score filter + NMS."""
	outputs = model([image_tensor.to(device)])
	prediction = {k: v.detach().cpu() for k, v in outputs[0].items()}
	prediction = apply_nms_to_prediction(
		prediction,
		iou_threshold=float(nms_iou_threshold),
		score_threshold=float(score_threshold),
	)
	return prediction


def draw_predictions(
	image_np: np.ndarray,
	prediction: Dict[str, torch.Tensor],
	class_names: List[str] | None = None,
) -> np.ndarray:
	"""Vẽ bbox + class + score lên ảnh, trả ảnh RGB numpy để hiển thị bằng Streamlit."""
	class_names = class_names or DEFAULT_CLASS_NAMES
	canvas = Image.fromarray(image_np)
	draw = ImageDraw.Draw(canvas)

	boxes = prediction.get("boxes", torch.empty((0, 4)))
	labels = prediction.get("labels", torch.empty((0,), dtype=torch.int64))
	scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32))

	for box, label, score in zip(boxes, labels, scores):
		x1, y1, x2, y2 = [int(v) for v in box.tolist()]

		# label của FasterRCNN: 1..num_classes, 0 là background
		class_idx = int(label.item()) - 1
		if 0 <= class_idx < len(class_names):
			class_name = class_names[class_idx]
		else:
			class_name = f"cls_{int(label.item())}"

		text = f"{class_name}: {float(score.item()):.2f}"
		draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
		draw.text((x1, max(0, y1 - 14)), text, fill=(0, 255, 0))

	return np.array(canvas)


@torch.no_grad()
def classify_boxes(
	image_np: np.ndarray,
	prediction: Dict[str, torch.Tensor],
	model: torch.nn.Module,
	device: str,
	idx_to_class: Dict[int, str],
) -> List[Dict[str, Any]]:
	"""Classify từng crop bbox và trả class/score theo thứ tự bbox detection."""
	results: List[Dict[str, Any]] = []
	boxes = prediction.get("boxes", torch.empty((0, 4)))

	for box in boxes:
		x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
		x1 = max(0, min(image_np.shape[1] - 1, x1))
		y1 = max(0, min(image_np.shape[0] - 1, y1))
		x2 = max(1, min(image_np.shape[1], x2))
		y2 = max(1, min(image_np.shape[0], y2))

		if x2 <= x1 or y2 <= y1:
			results.append({"class": "invalid_crop", "score": 0.0})
			continue

		crop = image_np[y1:y2, x1:x2]
		if crop.size == 0:
			results.append({"class": "invalid_crop", "score": 0.0})
			continue

		crop_img = Image.fromarray(crop).resize((224, 224), Image.BILINEAR)
		crop_np = np.array(crop_img).astype(np.float32) / 255.0
		x = torch.from_numpy(crop_np).permute(2, 0, 1).unsqueeze(0).to(device)

		logits = model(x)
		probs = torch.softmax(logits, dim=1)
		conf, pred_idx = torch.max(probs, dim=1)

		idx = int(pred_idx.item())
		results.append(
			{
				"class": idx_to_class.get(idx, f"cls_{idx}"),
				"score": float(conf.item()),
			}
		)

	return results


def fuse_detection_and_classification(
	det_class: str,
	det_score: float,
	cls_class: str,
	cls_score: float,
	cls_override_threshold: float = 0.65,
) -> Tuple[str, float, str]:
	"""Fusion rule đơn giản, dễ giải thích khi báo cáo."""
	if det_class == cls_class:
		final_score = max(det_score, cls_score)
		return det_class, final_score, "agree"

	if cls_score >= cls_override_threshold:
		return cls_class, cls_score, "cls_override"

	return det_class, det_score, "det_keep"


def build_interactive_bbox_figure(
	image_np: np.ndarray,
	rows: List[Dict[str, Any]],
):
	"""Tạo figure tương tác để hover lên bbox thấy class + score."""
	if not PLOTLY_AVAILABLE:
		return None

	height, width = image_np.shape[:2]
	fig = go.Figure()
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
		fig.add_trace(
			go.Scatter(
				x=[x1, x2, x2, x1, x1],
				y=[y1, y1, y2, y2, y1],
				mode="lines",
				line=dict(color="lime", width=2),
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
