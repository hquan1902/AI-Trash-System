import os
import time
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.common.data_pipeline import COCODetectionDataset, detection_collate_fn
from src.model_detect import build_detection_model
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR


def evaluate_map(
    model,
    loader,
    device,
    map_backend="faster_coco_eval",
    eval_score_th=None,
    eval_nms_iou_th=None,
    class_metrics=True,
    max_batches=None,
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
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
            metric.update(preds_cpu, targets_cpu)

            if max_batches is not None and batch_idx >= max_batches:
                break

    result = metric.compute()

    if hasattr(model, "roi_heads"):
        if old_score_th is not None:
            model.roi_heads.score_thresh = old_score_th
        if old_nms_iou_th is not None:
            model.roi_heads.nms_thresh = old_nms_iou_th

    return result


def check_class_coverage(coco_json_path, num_classes=7, split_name="val"):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    class_counts = Counter()
    for ann in coco.get("annotations", []):
        cid = int(ann["category_id"])
        class_counts[cid] += 1

    summary = {c: class_counts.get(c, 0) for c in range(num_classes)}
    missing = [c for c, n in summary.items() if n == 0]

    print(f"[{split_name}] class box counts: {summary}")
    if missing:
        print(f"[WARNING] Missing classes in {split_name}: {missing} -> per-class mAP có thể = -1")
    else:
        print(f"[{split_name}] OK: đủ {num_classes} lớp.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_batch_size = int(os.getenv("DET_BATCH_SIZE", "2"))
    val_batch_size = int(os.getenv("DET_VAL_BATCH_SIZE", "1"))
    num_workers = int(os.getenv("DET_NUM_WORKERS", "0"))
    epochs = int(os.getenv("DET_EPOCHS", "40"))
    map_backend = os.getenv("DET_MAP_BACKEND", "faster_coco_eval")

    # Các nút tuning nhanh (không cần sửa code lần sau)
    det_lr = float(os.getenv("DET_LR", "0.005"))
    det_momentum = float(os.getenv("DET_MOMENTUM", "0.9"))
    det_weight_decay = float(os.getenv("DET_WEIGHT_DECAY", "0.0005"))

    # Scheduler config
    det_scheduler = os.getenv("DET_SCHEDULER", "cosine").lower()
    det_step_size = int(os.getenv("DET_STEP_SIZE", "8"))
    det_gamma = float(os.getenv("DET_GAMMA", "0.1"))

    # Tách threshold cho evaluate và infer (fallback về DET_SCORE_TH để không phá lệnh cũ)
    legacy_score_th = os.getenv("DET_SCORE_TH", None)
    det_eval_score_th = float(os.getenv("DET_EVAL_SCORE_TH", legacy_score_th if legacy_score_th is not None else "0.001"))
    det_infer_score_th = float(os.getenv("DET_INFER_SCORE_TH", legacy_score_th if legacy_score_th is not None else "0.10"))
    det_nms_iou_th = float(os.getenv("DET_NMS_IOU_TH", "0.5"))

    # Scheduler mới: warmup + cosine
    det_warmup_epochs = int(os.getenv("DET_WARMUP_EPOCHS", "3"))
    det_min_lr = float(os.getenv("DET_MIN_LR", "1e-5"))

    # Tối ưu train ổn định
    det_use_amp = os.getenv("DET_USE_AMP", "1") == "1"
    det_grad_clip = float(os.getenv("DET_GRAD_CLIP", "5.0"))

    # Hard example mining nhẹ (sample-level reweight)
    det_use_hard_mining = os.getenv("DET_USE_HARD_MINING", "1") == "1"
    det_hard_alpha = float(os.getenv("DET_HARD_ALPHA", "0.35"))  # mức tăng trọng số ảnh khó

    # Speed knobs cho phase train: giảm tần suất/độ nặng evaluate
    det_eval_every = max(1, int(os.getenv("DET_EVAL_EVERY", "1")))
    det_eval_class_metrics = os.getenv("DET_EVAL_CLASS_METRICS", "1") == "1"
    det_val_max_batches = int(os.getenv("DET_VAL_MAX_BATCHES", "0"))
    det_full_eval_last = os.getenv("DET_FULL_EVAL_LAST", "1") == "1"

    # DataLoader speed knobs (hiệu quả khi num_workers > 0)
    det_prefetch_factor = max(1, int(os.getenv("DET_PREFETCH_FACTOR", "2")))
    det_persistent_workers = os.getenv("DET_PERSISTENT_WORKERS", "1") == "1"

    # Model init options
    det_use_pretrained = os.getenv("DET_USE_PRETRAINED", "0") == "1"
    det_model_variant = os.getenv("DET_MODEL_VARIANT", "v2").lower().strip()  # v1 | v2

    # Output options: tách cố định pretrained vs scratch
    # Chỉ cần đổi DET_USE_PRETRAINED=1/0 để chuyển nhánh chạy.
    det_pretrained_out_dir = os.getenv("DET_PRETRAINED_OUT_DIR", "models/pretrained")
    det_scratch_out_dir = os.getenv("DET_SCRATCH_OUT_DIR", "models")

    train_img_dir = os.getenv("DET_TRAIN_IMG_DIR", "data/detection_1/train")
    val_img_dir = os.getenv("DET_VAL_IMG_DIR", "data/detection_1/valid")
    train_json_path = os.getenv("DET_TRAIN_JSON", "data/detection_1/train/_annotations_7cls.coco.json")
    val_json_path = os.getenv("DET_VAL_JSON", "data/detection_1/valid/_annotations_7cls.coco.json")

    print(
        f"Detection config: train_bs={train_batch_size}, "
        f"val_bs={val_batch_size}, workers={num_workers}, epochs={epochs}, "
        f"map_backend={map_backend}, lr={det_lr}, momentum={det_momentum}, "
        f"weight_decay={det_weight_decay}, scheduler={det_scheduler}, "
        f"step_size={det_step_size}, gamma={det_gamma}, "
        f"eval_score_th={det_eval_score_th}, infer_score_th={det_infer_score_th}, "
        f"nms_iou_th={det_nms_iou_th}, "
        f"warmup_epochs={det_warmup_epochs}, min_lr={det_min_lr}, "
        f"use_amp={det_use_amp}, grad_clip={det_grad_clip}, "
        f"use_hard_mining={det_use_hard_mining}, hard_alpha={det_hard_alpha}, "
        f"eval_every={det_eval_every}, eval_class_metrics={det_eval_class_metrics}, "
        f"val_max_batches={det_val_max_batches}, full_eval_last={det_full_eval_last}, "
        f"use_pretrained={det_use_pretrained}, model_variant={det_model_variant}, "
        f"prefetch_factor={det_prefetch_factor}, "
        f"persistent_workers={det_persistent_workers}"
    )

    train_dataset = COCODetectionDataset(
        img_dir=train_img_dir,
        json_path=train_json_path,
        is_train=True,
        hflip_prob=0.5,
        color_jitter_prob=0.4,
    )
    val_dataset = COCODetectionDataset(
        img_dir=val_img_dir,
        json_path=val_json_path,
        is_train=False,
    )

    check_class_coverage(train_json_path, num_classes=7, split_name="train")
    check_class_coverage(val_json_path, num_classes=7, split_name="val")

    # Tính class weights từ train split để ưu tiên class hiếm
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    cls_counts = Counter()
    for ann in train_coco.get("annotations", []):
        cls_counts[int(ann["category_id"])] += 1

    # class_id trong annotation là 0..6; trong target labels của FasterRCNN là 1..7
    total_boxes = sum(cls_counts.values())
    cls_weights = {}
    for c in range(7):
        cnt = max(1, cls_counts.get(c, 1))
        cls_weights[c + 1] = total_boxes / (7.0 * cnt)

    print("Hard-mining class weights (label 1..7):", {k: round(v, 3) for k, v in cls_weights.items()})

    train_loader_kwargs = {}
    val_loader_kwargs = {}
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = det_persistent_workers
        val_loader_kwargs["persistent_workers"] = det_persistent_workers
        train_loader_kwargs["prefetch_factor"] = det_prefetch_factor
        val_loader_kwargs["prefetch_factor"] = det_prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device == "cuda"),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device == "cuda"),
        **val_loader_kwargs,
    )

    model = build_detection_model(
        num_classes=7,
        use_pretrained=det_use_pretrained,
        variant=det_model_variant,
    ).to(device)
    # Ngưỡng infer/demo mặc định của model (evaluate sẽ override bằng det_eval_score_th)
    model.roi_heads.score_thresh = det_infer_score_th
    model.roi_heads.nms_thresh = det_nms_iou_th

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=det_lr,
        momentum=det_momentum,
        weight_decay=det_weight_decay
    )

    if det_scheduler == "step":
        scheduler = StepLR(optimizer, step_size=det_step_size, gamma=det_gamma)
    else:
        # Cosine scheduler sau warmup (ổn định hơn StepLR cho train from scratch)
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - det_warmup_epochs),
            eta_min=det_min_lr
        )

        if det_warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.2,   # bắt đầu ở 20% LR
                end_factor=1.0,     # lên LR chuẩn
                total_iters=det_warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[det_warmup_epochs]
            )
        else:
            scheduler = main_scheduler

    scaler = GradScaler("cuda", enabled=(det_use_amp and device == "cuda"))

    best_map50 = -1.0
    best_map5095 = -1.0
    out_dir = Path(det_pretrained_out_dir if det_use_pretrained else det_scratch_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if det_use_pretrained:
        # Đường dẫn cố định riêng cho pretrained (đè theo lần chạy pretrained mới)
        last_ckpt_path = out_dir / "model_detection_pretrained_last.pth"
        best50_ckpt_path = out_dir / "model_detection_pretrained_best_map50_29_4.pth"
        best5095_ckpt_path = out_dir / "model_detection_pretrained_best_map5095_29_4.pth"
    else:
        # Đường dẫn mặc định nhánh scratch (giữ tương thích toàn bộ luồng cũ)
        last_ckpt_path = out_dir / "model_detection_last.pth"
        best50_ckpt_path = out_dir / "model_detection_best_7_4.pth"
        best5095_ckpt_path = out_dir / "model_detection_best_map5095_7_4.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_phase_start = time.time()
        running_loss = 0.0
        loss_cls_sum = 0.0
        loss_box_sum = 0.0
        loss_obj_sum = 0.0
        loss_rpn_sum = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(det_use_amp and device == "cuda")):
                loss_dict = model(images, targets)
                base_loss = sum(loss_dict.values())

                if det_use_hard_mining:
                    # Tính độ khó trung bình của batch từ các labels xuất hiện
                    batch_label_set = set()
                    for t in targets:
                        if "labels" in t and t["labels"].numel() > 0:
                            for lb in t["labels"].detach().cpu().tolist():
                                batch_label_set.add(int(lb))  # labels đang là 1..7

                    if len(batch_label_set) > 0:
                        batch_w = sum(cls_weights.get(lb, 1.0) for lb in batch_label_set) / len(batch_label_set)
                    else:
                        batch_w = 1.0

                    # scale nhẹ để tránh làm mất ổn định
                    loss = base_loss * (1.0 + det_hard_alpha * (batch_w - 1.0))
                else:
                    loss = base_loss

            loss_cls_sum += float(loss_dict["loss_classifier"].item())
            loss_box_sum += float(loss_dict["loss_box_reg"].item())
            loss_obj_sum += float(loss_dict["loss_objectness"].item())
            loss_rpn_sum += float(loss_dict["loss_rpn_box_reg"].item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # cần trước clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=det_grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        num_steps = max(1, len(train_loader))
        loss_cls = loss_cls_sum / num_steps
        loss_box = loss_box_sum / num_steps
        loss_obj = loss_obj_sum / num_steps
        loss_rpn = loss_rpn_sum / num_steps
        train_phase_time = time.time() - train_phase_start

        train_loss = running_loss / num_steps
        should_eval = (epoch % det_eval_every == 0) or (epoch == epochs)
        force_full_eval = (epoch == epochs) and det_full_eval_last
        if should_eval:
            eval_class_metrics = True if force_full_eval else det_eval_class_metrics
            eval_max_batches = None if force_full_eval else (None if det_val_max_batches <= 0 else det_val_max_batches)

            eval_phase_start = time.time()
            map_result = evaluate_map(
                model,
                val_loader,
                device,
                map_backend=map_backend,
                eval_score_th=det_eval_score_th,
                eval_nms_iou_th=det_nms_iou_th,
                class_metrics=eval_class_metrics,
                max_batches=eval_max_batches,
            )
            eval_phase_time = time.time() - eval_phase_start

            map_per_class = map_result.get("map_per_class", None)
            classes = map_result.get("classes", None)
            if eval_class_metrics and map_per_class is not None and classes is not None:
                print("  per-class mAP:")
                for cls_id, cls_map in zip(classes.tolist(), map_per_class.tolist()):
                    print(f"    class_id={cls_id}: mAP={cls_map:.4f}")

            map50 = float(map_result["map_50"])
            map5095 = float(map_result["map"])
        else:
            map50 = float("nan")
            map5095 = float("nan")
            eval_phase_time = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | map50={map50:.4f} | map50_95={map5095:.4f} | "
            f"loss_cls={loss_cls:.4f} | loss_box={loss_box:.4f} | "
            f"loss_obj={loss_obj:.4f} | loss_rpn={loss_rpn:.4f} | "
            f"lr={current_lr:.6f} | time={epoch_time:.1f}s | train_t={train_phase_time:.1f}s | "
            f"eval_t={eval_phase_time:.1f}s | eval={'yes' if should_eval else 'no'}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "map50": map50,
            "map50_95": map5095,
            "use_pretrained": det_use_pretrained,
            "model_variant": det_model_variant,
            "train_json_path": train_json_path,
            "val_json_path": val_json_path,
        }

        torch.save(ckpt_payload, str(last_ckpt_path))

        if should_eval and map50 > best_map50:
            best_map50 = map50
            torch.save(ckpt_payload, str(best50_ckpt_path))
            print(f"  -> New BEST detection saved (map50={best_map50:.4f})")

        if should_eval and map5095 > best_map5095:
            best_map5095 = map5095
            torch.save(ckpt_payload, str(best5095_ckpt_path))
            print(f"  -> New BEST (map50_95={best5095:.4f})")

        scheduler.step()

    print("Detection training finished.")
    print(f"Saved: {last_ckpt_path.as_posix()}")
    print(f"Saved: {best50_ckpt_path.as_posix()}")
    print(f"Saved: {best5095_ckpt_path.as_posix()}")


if __name__ == "__main__":
    main()"""Compatibility entrypoint for detection training."""

from src.detection.train import main


if __name__ == "__main__":
    main()import os
import time
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.data_pipeline import COCODetectionDataset, detection_collate_fn
from src.model_detect import build_detection_model
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR


def evaluate_map(
    model,
    loader,
    device,
    map_backend="faster_coco_eval",
    eval_score_th=None,
    eval_nms_iou_th=None,
    class_metrics=True,
    max_batches=None,
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
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
            metric.update(preds_cpu, targets_cpu)

            if max_batches is not None and batch_idx >= max_batches:
                break

    result = metric.compute()

    if hasattr(model, "roi_heads"):
        if old_score_th is not None:
            model.roi_heads.score_thresh = old_score_th
        if old_nms_iou_th is not None:
            model.roi_heads.nms_thresh = old_nms_iou_th

    return result


def check_class_coverage(coco_json_path, num_classes=7, split_name="val"):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    class_counts = Counter()
    for ann in coco.get("annotations", []):
        cid = int(ann["category_id"])
        class_counts[cid] += 1

    summary = {c: class_counts.get(c, 0) for c in range(num_classes)}
    missing = [c for c, n in summary.items() if n == 0]

    print(f"[{split_name}] class box counts: {summary}")
    if missing:
        print(f"[WARNING] Missing classes in {split_name}: {missing} -> per-class mAP có thể = -1")
    else:
        print(f"[{split_name}] OK: đủ {num_classes} lớp.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_batch_size = int(os.getenv("DET_BATCH_SIZE", "2"))
    val_batch_size = int(os.getenv("DET_VAL_BATCH_SIZE", "1"))
    num_workers = int(os.getenv("DET_NUM_WORKERS", "0"))
    epochs = int(os.getenv("DET_EPOCHS", "40"))
    map_backend = os.getenv("DET_MAP_BACKEND", "faster_coco_eval")

    # Các nút tuning nhanh (không cần sửa code lần sau)
    det_lr = float(os.getenv("DET_LR", "0.005"))
    det_momentum = float(os.getenv("DET_MOMENTUM", "0.9"))
    det_weight_decay = float(os.getenv("DET_WEIGHT_DECAY", "0.0005"))

    # Scheduler config
    det_scheduler = os.getenv("DET_SCHEDULER", "cosine").lower()
    det_step_size = int(os.getenv("DET_STEP_SIZE", "8"))
    det_gamma = float(os.getenv("DET_GAMMA", "0.1"))

    # Tách threshold cho evaluate và infer (fallback về DET_SCORE_TH để không phá lệnh cũ)
    legacy_score_th = os.getenv("DET_SCORE_TH", None)
    det_eval_score_th = float(os.getenv("DET_EVAL_SCORE_TH", legacy_score_th if legacy_score_th is not None else "0.001"))
    det_infer_score_th = float(os.getenv("DET_INFER_SCORE_TH", legacy_score_th if legacy_score_th is not None else "0.10"))
    det_nms_iou_th = float(os.getenv("DET_NMS_IOU_TH", "0.5"))

    # Scheduler mới: warmup + cosine
    det_warmup_epochs = int(os.getenv("DET_WARMUP_EPOCHS", "3"))
    det_min_lr = float(os.getenv("DET_MIN_LR", "1e-5"))

    # Tối ưu train ổn định
    det_use_amp = os.getenv("DET_USE_AMP", "1") == "1"
    det_grad_clip = float(os.getenv("DET_GRAD_CLIP", "5.0"))

    # Hard example mining nhẹ (sample-level reweight)
    det_use_hard_mining = os.getenv("DET_USE_HARD_MINING", "1") == "1"
    det_hard_alpha = float(os.getenv("DET_HARD_ALPHA", "0.35"))  # mức tăng trọng số ảnh khó

    # Speed knobs cho phase train: giảm tần suất/độ nặng evaluate
    det_eval_every = max(1, int(os.getenv("DET_EVAL_EVERY", "1")))
    det_eval_class_metrics = os.getenv("DET_EVAL_CLASS_METRICS", "1") == "1"
    det_val_max_batches = int(os.getenv("DET_VAL_MAX_BATCHES", "0"))
    det_full_eval_last = os.getenv("DET_FULL_EVAL_LAST", "1") == "1"

    # DataLoader speed knobs (hiệu quả khi num_workers > 0)
    det_prefetch_factor = max(1, int(os.getenv("DET_PREFETCH_FACTOR", "2")))
    det_persistent_workers = os.getenv("DET_PERSISTENT_WORKERS", "1") == "1"

    # Model init options
    det_use_pretrained = os.getenv("DET_USE_PRETRAINED", "0") == "1"
    det_model_variant = os.getenv("DET_MODEL_VARIANT", "v2").lower().strip()  # v1 | v2

    # Output options: tách cố định pretrained vs scratch
    # Chỉ cần đổi DET_USE_PRETRAINED=1/0 để chuyển nhánh chạy.
    det_pretrained_out_dir = os.getenv("DET_PRETRAINED_OUT_DIR", "models/pretrained")
    det_scratch_out_dir = os.getenv("DET_SCRATCH_OUT_DIR", "models")

    train_img_dir = os.getenv("DET_TRAIN_IMG_DIR", "data/detection_1/train")
    val_img_dir = os.getenv("DET_VAL_IMG_DIR", "data/detection_1/valid")
    train_json_path = os.getenv("DET_TRAIN_JSON", "data/detection_1/train/_annotations_7cls.coco.json")
    val_json_path = os.getenv("DET_VAL_JSON", "data/detection_1/valid/_annotations_7cls.coco.json")

    print(
        f"Detection config: train_bs={train_batch_size}, "
        f"val_bs={val_batch_size}, workers={num_workers}, epochs={epochs}, "
        f"map_backend={map_backend}, lr={det_lr}, momentum={det_momentum}, "
        f"weight_decay={det_weight_decay}, scheduler={det_scheduler}, "
        f"step_size={det_step_size}, gamma={det_gamma}, "
        f"eval_score_th={det_eval_score_th}, infer_score_th={det_infer_score_th}, "
        f"nms_iou_th={det_nms_iou_th}, "
        f"warmup_epochs={det_warmup_epochs}, min_lr={det_min_lr}, "
        f"use_amp={det_use_amp}, grad_clip={det_grad_clip}, "
        f"use_hard_mining={det_use_hard_mining}, hard_alpha={det_hard_alpha}, "
        f"eval_every={det_eval_every}, eval_class_metrics={det_eval_class_metrics}, "
        f"val_max_batches={det_val_max_batches}, full_eval_last={det_full_eval_last}, "
        f"use_pretrained={det_use_pretrained}, model_variant={det_model_variant}, "
        f"prefetch_factor={det_prefetch_factor}, "
        f"persistent_workers={det_persistent_workers}"
    )

    train_dataset = COCODetectionDataset(
        img_dir=train_img_dir,
        json_path=train_json_path,
        is_train=True,
        hflip_prob=0.5,
        color_jitter_prob=0.4,
    )
    val_dataset = COCODetectionDataset(
        img_dir=val_img_dir,
        json_path=val_json_path,
        is_train=False,
    )

    check_class_coverage(train_json_path, num_classes=7, split_name="train")
    check_class_coverage(val_json_path, num_classes=7, split_name="val")

    # Tính class weights từ train split để ưu tiên class hiếm
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    cls_counts = Counter()
    for ann in train_coco.get("annotations", []):
        cls_counts[int(ann["category_id"])] += 1

    # class_id trong annotation là 0..6; trong target labels của FasterRCNN là 1..7
    total_boxes = sum(cls_counts.values())
    cls_weights = {}
    for c in range(7):
        cnt = max(1, cls_counts.get(c, 1))
        cls_weights[c + 1] = total_boxes / (7.0 * cnt)

    print("Hard-mining class weights (label 1..7):", {k: round(v, 3) for k, v in cls_weights.items()})

    train_loader_kwargs = {}
    val_loader_kwargs = {}
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = det_persistent_workers
        val_loader_kwargs["persistent_workers"] = det_persistent_workers
        train_loader_kwargs["prefetch_factor"] = det_prefetch_factor
        val_loader_kwargs["prefetch_factor"] = det_prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device == "cuda"),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device == "cuda"),
        **val_loader_kwargs,
    )

    model = build_detection_model(
        num_classes=7,
        use_pretrained=det_use_pretrained,
        variant=det_model_variant,
    ).to(device)
    # Ngưỡng infer/demo mặc định của model (evaluate sẽ override bằng det_eval_score_th)
    model.roi_heads.score_thresh = det_infer_score_th
    model.roi_heads.nms_thresh = det_nms_iou_th

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=det_lr,
        momentum=det_momentum,
        weight_decay=det_weight_decay
    )

    if det_scheduler == "step":
        scheduler = StepLR(optimizer, step_size=det_step_size, gamma=det_gamma)
    else:
        # Cosine scheduler sau warmup (ổn định hơn StepLR cho train from scratch)
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - det_warmup_epochs),
            eta_min=det_min_lr
        )

        if det_warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.2,   # bắt đầu ở 20% LR
                end_factor=1.0,     # lên LR chuẩn
                total_iters=det_warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[det_warmup_epochs]
            )
        else:
            scheduler = main_scheduler

    scaler = GradScaler("cuda", enabled=(det_use_amp and device == "cuda"))

    best_map50 = -1.0
    best_map5095 = -1.0
    out_dir = Path(det_pretrained_out_dir if det_use_pretrained else det_scratch_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if det_use_pretrained:
        # Đường dẫn cố định riêng cho pretrained (đè theo lần chạy pretrained mới)
        last_ckpt_path = out_dir / "model_detection_pretrained_last.pth"
        best50_ckpt_path = out_dir / "model_detection_pretrained_best_map50_29_4.pth"
        best5095_ckpt_path = out_dir / "model_detection_pretrained_best_map5095_29_4.pth"
    else:
        # Đường dẫn mặc định nhánh scratch (giữ tương thích toàn bộ luồng cũ)
        last_ckpt_path = out_dir / "model_detection_last.pth"
        best50_ckpt_path = out_dir / "model_detection_best_7_4.pth"
        best5095_ckpt_path = out_dir / "model_detection_best_map5095_7_4.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_phase_start = time.time()
        running_loss = 0.0
        loss_cls_sum = 0.0
        loss_box_sum = 0.0
        loss_obj_sum = 0.0
        loss_rpn_sum = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(det_use_amp and device == "cuda")):
                loss_dict = model(images, targets)
                base_loss = sum(loss_dict.values())

                if det_use_hard_mining:
                    # Tính độ khó trung bình của batch từ các labels xuất hiện
                    batch_label_set = set()
                    for t in targets:
                        if "labels" in t and t["labels"].numel() > 0:
                            for lb in t["labels"].detach().cpu().tolist():
                                batch_label_set.add(int(lb))  # labels đang là 1..7

                    if len(batch_label_set) > 0:
                        batch_w = sum(cls_weights.get(lb, 1.0) for lb in batch_label_set) / len(batch_label_set)
                    else:
                        batch_w = 1.0

                    # scale nhẹ để tránh làm mất ổn định
                    loss = base_loss * (1.0 + det_hard_alpha * (batch_w - 1.0))
                else:
                    loss = base_loss

            loss_cls_sum += float(loss_dict["loss_classifier"].item())
            loss_box_sum += float(loss_dict["loss_box_reg"].item())
            loss_obj_sum += float(loss_dict["loss_objectness"].item())
            loss_rpn_sum += float(loss_dict["loss_rpn_box_reg"].item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # cần trước clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=det_grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        num_steps = max(1, len(train_loader))
        loss_cls = loss_cls_sum / num_steps
        loss_box = loss_box_sum / num_steps
        loss_obj = loss_obj_sum / num_steps
        loss_rpn = loss_rpn_sum / num_steps
        train_phase_time = time.time() - train_phase_start

        train_loss = running_loss / num_steps
        should_eval = (epoch % det_eval_every == 0) or (epoch == epochs)
        force_full_eval = (epoch == epochs) and det_full_eval_last
        if should_eval:
            eval_class_metrics = True if force_full_eval else det_eval_class_metrics
            eval_max_batches = None if force_full_eval else (None if det_val_max_batches <= 0 else det_val_max_batches)

            eval_phase_start = time.time()
            map_result = evaluate_map(
                model,
                val_loader,
                device,
                map_backend=map_backend,
                eval_score_th=det_eval_score_th,
                eval_nms_iou_th=det_nms_iou_th,
                class_metrics=eval_class_metrics,
                max_batches=eval_max_batches,
            )
            eval_phase_time = time.time() - eval_phase_start

            map_per_class = map_result.get("map_per_class", None)
            classes = map_result.get("classes", None)
            if eval_class_metrics and map_per_class is not None and classes is not None:
                print("  per-class mAP:")
                for cls_id, cls_map in zip(classes.tolist(), map_per_class.tolist()):
                    print(f"    class_id={cls_id}: mAP={cls_map:.4f}")

            map50 = float(map_result["map_50"])
            map5095 = float(map_result["map"])
        else:
            map50 = float("nan")
            map5095 = float("nan")
            eval_phase_time = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | map50={map50:.4f} | map50_95={map5095:.4f} | "
            f"loss_cls={loss_cls:.4f} | loss_box={loss_box:.4f} | "
            f"loss_obj={loss_obj:.4f} | loss_rpn={loss_rpn:.4f} | "
            f"lr={current_lr:.6f} | time={epoch_time:.1f}s | train_t={train_phase_time:.1f}s | "
            f"eval_t={eval_phase_time:.1f}s | eval={'yes' if should_eval else 'no'}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "map50": map50,
            "map50_95": map5095,
            "use_pretrained": det_use_pretrained,
            "model_variant": det_model_variant,
            "train_json_path": train_json_path,
            "val_json_path": val_json_path,
        }

        torch.save(ckpt_payload, str(last_ckpt_path))

        if should_eval and map50 > best_map50:
            best_map50 = map50
            torch.save(ckpt_payload, str(best50_ckpt_path))
            print(f"  -> New BEST detection saved (map50={best_map50:.4f})")

        if should_eval and map5095 > best_map5095:
            best_map5095 = map5095
            torch.save(ckpt_payload, str(best5095_ckpt_path))
            print(f"  -> New BEST (map50_95={best_map5095:.4f})")

        scheduler.step()

    print("Detection training finished.")
    print(f"Saved: {last_ckpt_path.as_posix()}")
    print(f"Saved: {best50_ckpt_path.as_posix()}")
    print(f"Saved: {best5095_ckpt_path.as_posix()}")


if __name__ == "__main__":
    main()

# ---------------- QUICK RUN COMMANDS (PowerShell) ----------------
# Full-run pretrained on detection_merged_bio:

# $env:DET_TRAIN_IMG_DIR="data"
# $env:DET_VAL_IMG_DIR="data"
# $env:DET_TRAIN_JSON="data/detection_merged_bio/train/_annotations_7cls_split_train.coco.json"
# $env:DET_VAL_JSON="data/detection_merged_bio/valid/_annotations_7cls_split_val.coco.json"
#
# $env:DET_USE_PRETRAINED="1"
# $env:DET_MODEL_VARIANT="v2"
# $env:DET_PRETRAINED_OUT_DIR="models/pretrained"
#
# $env:DET_BATCH_SIZE="6"
# $env:DET_VAL_BATCH_SIZE="3"
# $env:DET_NUM_WORKERS="4"
# $env:DET_PREFETCH_FACTOR="4"
# $env:DET_PERSISTENT_WORKERS="1"
#
# $env:DET_EPOCHS="50"
# $env:DET_MAP_BACKEND="faster_coco_eval"
# $env:DET_LR="0.0015"
# $env:DET_MOMENTUM="0.9"
# $env:DET_WEIGHT_DECAY="0.0005"
# $env:DET_SCHEDULER="cosine"
# $env:DET_WARMUP_EPOCHS="2"
# $env:DET_MIN_LR="1e-5"
#
# $env:DET_USE_AMP="1"
# $env:DET_GRAD_CLIP="5.0"
# $env:DET_USE_HARD_MINING="1"
# $env:DET_HARD_ALPHA="0.25"
#
# $env:DET_EVAL_SCORE_TH="0.001"
# $env:DET_INFER_SCORE_TH="0.10"
# $env:DET_NMS_IOU_TH="0.5"
#
# $env:DET_EVAL_EVERY="1"
# $env:DET_EVAL_CLASS_METRICS="1"
# $env:DET_VAL_MAX_BATCHES="0"
# $env:DET_FULL_EVAL_LAST="1"
#
# python -m src.train_detect



# Full-run scratch on detection_merged_bio (switch back):

# $env:DET_TRAIN_IMG_DIR="data"
# $env:DET_VAL_IMG_DIR="data"
# $env:DET_TRAIN_JSON="data/detection_merged_bio/train/_annotations_7cls_split_train.coco.json"
# $env:DET_VAL_JSON="data/detection_merged_bio/valid/_annotations_7cls_split_val.coco.json"
#
# $env:DET_USE_PRETRAINED="0"
# $env:DET_MODEL_VARIANT="v2"
# $env:DET_SCRATCH_OUT_DIR="models"
#
# $env:DET_BATCH_SIZE="6"
# $env:DET_VAL_BATCH_SIZE="3"
# $env:DET_NUM_WORKERS="4"
# $env:DET_PREFETCH_FACTOR="4"
# $env:DET_PERSISTENT_WORKERS="1"
#
# $env:DET_EPOCHS="50"
# $env:DET_MAP_BACKEND="faster_coco_eval"
# $env:DET_LR="0.005"
# $env:DET_MOMENTUM="0.9"
# $env:DET_WEIGHT_DECAY="0.0005"
# $env:DET_SCHEDULER="cosine"
# $env:DET_WARMUP_EPOCHS="3"
# $env:DET_MIN_LR="1e-5"
#
# $env:DET_USE_AMP="1"
# $env:DET_GRAD_CLIP="5.0"
# $env:DET_USE_HARD_MINING="1"
# $env:DET_HARD_ALPHA="0.35"
#
# $env:DET_EVAL_SCORE_TH="0.001"
# $env:DET_INFER_SCORE_TH="0.10"
# $env:DET_NMS_IOU_TH="0.5"
#
# $env:DET_EVAL_EVERY="1"
# $env:DET_EVAL_CLASS_METRICS="1"
# $env:DET_VAL_MAX_BATCHES="0"
# $env:DET_FULL_EVAL_LAST="1"
#
# python -m src.train_detect

# Optional high-VRAM profile (pretrained):
# $env:DET_BATCH_SIZE="6"
# $env:DET_VAL_BATCH_SIZE="3"
# $env:DET_NUM_WORKERS="4"
# $env:DET_PREFETCH_FACTOR="4"