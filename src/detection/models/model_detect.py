import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms


def build_detection_model(num_classes: int, use_pretrained: bool = False, variant: str = "v2"):
    # num_classes(7): số class không tính background
    # variant: "v1" -> fasterrcnn_resnet50_fpn, "v2" -> fasterrcnn_resnet50_fpn_v2

    variant = str(variant).lower().strip()

    if variant == "v1":
        if use_pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
            )
    else:
        if use_pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
            )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 background
    return model


def apply_nms_to_prediction(prediction, iou_threshold=0.5, score_threshold=0.05):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    keep_idx = nms(boxes, scores, iou_threshold)
    return {
        "boxes": boxes[keep_idx],
        "scores": scores[keep_idx],
        "labels": labels[keep_idx],
    }
