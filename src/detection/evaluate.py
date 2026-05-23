import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.common.data_pipeline import COCODetectionDataset, detection_collate_fn
from src.detection.models.model_detect import build_detection_model


def evaluate_map(
	model,
	loader,
	device,
	map_backend="faster_coco_eval",
	eval_score_th=None,
	eval_nms_iou_th=None,
	class_metrics=True,
):
	metric = MeanAveragePrecision(iou_type="bbox", backend=map_backend, class_metrics=class_metrics)
	model.eval()

	old_score_th = None
	old_nms_iou_th = None
	if hasattr(model, "roi_heads"):
		old_score_th = getattr(model.roi_heads, "score_thresh", None)
		old_nms_iou_th = getattr(model.roi_heads, "nms_thresh", None)
		if eval_score_th is not None:
			model.roi_heads.score_thresh = float(eval_score_th)
		if eval_nms_iou_th is not None:
			model.roi_heads.nms_thresh = float(eval_nms_iou_th)

	with torch.no_grad():
		for images, targets in loader:
			images = [img.to(device) for img in images]
			outputs = model(images)

			preds_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
			targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
			metric.update(preds_cpu, targets_cpu)

	result = metric.compute()

	if hasattr(model, "roi_heads"):
		if old_score_th is not None:
			model.roi_heads.score_thresh = old_score_th
		if old_nms_iou_th is not None:
			model.roi_heads.nms_thresh = old_nms_iou_th

	return result


def main():
	parser = argparse.ArgumentParser(description="Evaluate detection model on COCO val set")
	parser.add_argument("--ckpt", default="models/model_detection_last.pth", help="Checkpoint path")
	parser.add_argument("--train-img-dir", default=os.getenv("DET_TRAIN_IMG_DIR", "data/detection_1/train"))
	parser.add_argument("--val-img-dir", default=os.getenv("DET_VAL_IMG_DIR", "data/detection_1/valid"))
	parser.add_argument(
		"--val-json",
		default=os.getenv("DET_VAL_JSON", "data/detection_1/valid/_annotations_7cls.coco.json"),
	)
	parser.add_argument("--batch-size", type=int, default=int(os.getenv("DET_VAL_BATCH_SIZE", "1")))
	parser.add_argument("--num-workers", type=int, default=int(os.getenv("DET_NUM_WORKERS", "0")))
	parser.add_argument("--map-backend", default=os.getenv("DET_MAP_BACKEND", "faster_coco_eval"))
	parser.add_argument("--eval-score-th", type=float, default=float(os.getenv("DET_EVAL_SCORE_TH", "0.001")))
	parser.add_argument("--nms-iou-th", type=float, default=float(os.getenv("DET_NMS_IOU_TH", "0.5")))
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Device: {device}")

	ckpt = torch.load(args.ckpt, map_location=device)
	use_pretrained = ckpt.get("use_pretrained", os.getenv("DET_USE_PRETRAINED", "0") == "1")
	model_variant = ckpt.get("model_variant", os.getenv("DET_MODEL_VARIANT", "v2")).lower()

	model = build_detection_model(num_classes=7, use_pretrained=use_pretrained, variant=model_variant).to(device)
	model.load_state_dict(ckpt["model_state_dict"])

	val_dataset = COCODetectionDataset(
		img_dir=args.val_img_dir,
		json_path=args.val_json,
		is_train=False,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=detection_collate_fn,
		pin_memory=(device == "cuda"),
	)

	result = evaluate_map(
		model,
		val_loader,
		device,
		map_backend=args.map_backend,
		eval_score_th=args.eval_score_th,
		eval_nms_iou_th=args.nms_iou_th,
		class_metrics=True,
	)

	print("mAP50:", float(result["map_50"]))
	print("mAP50-95:", float(result["map"]))


if __name__ == "__main__":
	main()
