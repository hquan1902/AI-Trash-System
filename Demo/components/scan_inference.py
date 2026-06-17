import os
import time
import hashlib
import io
import streamlit as st
import torch
import numpy as np
from PIL import Image
from datetime import datetime

# Imports from Demo.utils
from Demo.utils import (
    predict_classification,
    run_detection,
    classify_crops,
    fuse_detection_and_classification,
    align_class_name,
    USER_CLASS_NAMES,
    DET_CLASS_NAMES
)

def save_corrected_image(image_input: Image.Image, entry: dict, corrected_class: str):
    """
    Saves the corrected image (full image for classification, or crop for detection)
    to Trash_img/{corrected_class}/ for future training.
    """
    try:
        # Determine output folder
        folder_path = os.path.join("Trash_img", corrected_class)
        os.makedirs(folder_path, exist_ok=True)
        
        # Parse timestamp from ID
        img_key = entry["image_key"]
        timestamp = entry["id"].split("_")[-1]
        
        if entry["type"] == "classification":
            filename = f"{img_key}_{timestamp}.png"
            file_path = os.path.join(folder_path, filename)
            image_input.save(file_path, format="PNG")
        else:
            # Crop bounding box area
            x1, y1, x2, y2 = [int(v) for v in entry["box"]]
            image_np = np.array(image_input)
            crop_np = image_np[y1:y2, x1:x2]
            
            if crop_np.size > 0:
                crop_pil = Image.fromarray(crop_np)
                # Parse box index
                box_idx = entry["id"].split("_box_")[-1].split("_")[0]
                filename = f"{img_key}_box_{box_idx}_{timestamp}.png"
                file_path = os.path.join(folder_path, filename)
                crop_pil.save(file_path, format="PNG")
                
        st.toast(f"💾 Đã lưu ảnh sửa đổi vào thư mục Trash_img/{corrected_class}/", icon="💾")
    except Exception as e:
        st.error(f"⚠️ Không thể lưu ảnh sửa đổi vào bộ dữ liệu: {e}")

def run_inference_if_new_state(
    classifier_model,
    detector_model,
    image_input,
    img_key,
    current_state_key,
    app_mode,
    detector_path,
    classifier_choice,
    score_threshold,
    nms_iou_threshold,
    cls_override_threshold
):
    """Runs AI inference if settings or image changed, and populates session history."""
    if st.session_state.last_processed_key != current_state_key:
        with st.spinner("Đang xử lý ảnh bằng AI..."):
            time.sleep(0.2)
            time_str = datetime.now().strftime("%H:%M:%S")
            date_str = datetime.now().strftime("%d/%m/%Y")
            scan_time = int(time.time() * 1000)
            
            if app_mode == "Phân loại ảnh đơn (ResNet50)":
                probs, top3 = predict_classification(image_input, classifier_model)
                label = USER_CLASS_NAMES[top3[0]]
                conf = probs[top3[0]] * 100
                new_id = f"cls_{img_key}_{scan_time}"
                
                st.session_state.history.append({
                    "id": new_id,
                    "type": "classification",
                    "image_key": img_key,
                    "label": label,
                    "corrected": None,
                    "is_correct": None,
                    "conf": conf,
                    "time": time_str,
                    "date": date_str,
                    "probs": probs.tolist(),
                    "top3": top3,
                })
                st.session_state.active_ids = [new_id]
            else:
                # Run Faster R-CNN detection
                prediction = run_detection(
                    model=detector_model,
                    pil_image=image_input,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold
                )
                
                boxes = prediction.get("boxes", torch.empty((0, 4)))
                labels = prediction.get("labels", torch.empty((0,), dtype=torch.int64))
                scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32))
                num_boxes = int(boxes.shape[0])
                
                if num_boxes > 0:
                    image_np = np.array(image_input)
                    # Check if we should classify using ResNet50
                    use_resnet = (
                        app_mode == "Phát hiện vật thể + Phân loại" and 
                        classifier_choice == "Mô hình phân loại (ResNet50)"
                    )
                    if use_resnet:
                        cls_results = classify_crops(image_np, prediction, classifier_model)
                    else:
                        cls_results = []
                        
                    new_active_ids = []
                    for idx, (box, label_tensor, score_tensor) in enumerate(zip(boxes, labels, scores)):
                        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                        det_idx = int(label_tensor.item()) - 1
                        det_class = DET_CLASS_NAMES[det_idx] if 0 <= det_idx < len(DET_CLASS_NAMES) else f"cls_{int(label_tensor.item())}"
                        det_score = float(score_tensor.item())
                        
                        det_class_aligned = align_class_name(det_class)
                        
                        if use_resnet and idx < len(cls_results):
                            cls_class = cls_results[idx]["class"]
                            cls_score = cls_results[idx]["score"]
                            cls_probs = cls_results[idx]["probs"].tolist()
                            
                            final_class, final_score, fusion_source = fuse_detection_and_classification(
                                det_class=det_class,
                                det_score=det_score,
                                cls_class=cls_class,
                                cls_score=cls_score,
                                cls_override_threshold=cls_override_threshold
                            )
                        else:
                            cls_class = None
                            cls_score = 0.0
                            cls_probs = []
                            final_class, final_score, fusion_source = det_class_aligned, det_score, "det_only"
                            
                        new_id = f"det_{img_key}_box_{idx}_{scan_time}"
                        st.session_state.history.append({
                            "id": new_id,
                            "type": "detection",
                            "image_key": img_key,
                            "box": [x1, y1, x2, y2],
                            "det_class": det_class_aligned,
                            "det_score": det_score * 100,
                            "cls_class": cls_class,
                            "cls_score": cls_score * 100,
                            "label": final_class,
                            "corrected": None,
                            "is_correct": None,
                            "conf": final_score * 100,
                            "fusion_source": fusion_source,
                            "time": time_str,
                            "date": date_str,
                            "cls_probs": cls_probs,
                        })
                        new_active_ids.append(new_id)
                    st.session_state.active_ids = new_active_ids
                else:
                    st.session_state.active_ids = []
                            
        st.session_state.last_processed_key = current_state_key
