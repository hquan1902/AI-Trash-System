import streamlit as st
import numpy as np
from collections import Counter
from Demo.utils import CLASS_INFO

def render_sidebar(history):
    """
    Render ECOSORT AI sidebar options and session statistics.
    Returns:
        tuple: (app_mode, score_threshold, nms_iou_threshold, cls_override_threshold)
    """
    with st.sidebar:
        # App Header Logo & Titles
        st.markdown("""
        <div style='padding:1.4rem 0 1rem; text-align:center;'>
            <div style='font-size:2.2rem'>♻️</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:1.1rem;
                        font-weight:500; color:#8ACA8A; letter-spacing:1px; margin-top:0.4rem;'>
                ECOSORT AI
            </div>
            <div style='font-size:0.72rem; color:#4A6A4A; margin-top:0.2rem;
                        letter-spacing:0.08em; text-transform:uppercase;'>
                ResNet50 + Faster R-CNN
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        st.markdown("<div class='section-title'>Cấu hình suy luận</div>", unsafe_allow_html=True)
        
        # Selection of Mode
        app_mode = st.selectbox(
            "Chế độ hoạt động",
            options=[
                "Phân loại ảnh đơn (ResNet50)",
                "Chỉ phát hiện vật thể",
                "Phát hiện vật thể + Phân loại"
            ],
            index=2  # Combined mode by default
        )
        
        # Sub-selection for detection model
        import os
        detector_path = "models/detection_pretrained/model_detection_pretrained_best_map50.pth"
        if not os.path.exists(detector_path):
            detector_path = "models/model_detection_pretrained_best_map50.pth"
            
        if app_mode != "Phân loại ảnh đơn (ResNet50)":
            detector_choice = st.selectbox(
                "Chọn mô hình phát hiện",
                options=[
                    "Mô hình pre-trained (model_detection_pretrained_best_map50.pth)",
                    "Mô hình tự train (model_detection_best_7_4.pth)"
                ],
                index=0,
                help="Chọn phiên bản mô hình Faster R-CNN để phát hiện vật thể."
            )
            if "model_detection_pretrained_best_map50.pth" in detector_choice:
                detector_path = "models/detection_pretrained/model_detection_pretrained_best_map50.pth"
                if not os.path.exists(detector_path):
                    detector_path = "models/model_detection_pretrained_best_map50.pth"
            else:
                detector_path = "models/detection_train_from_scratch/model_detection_best_7_4.pth"
                if not os.path.exists(detector_path):
                    detector_path = "models/model_detection_best_7_4.pth"
        
        # Sub-selection for classification model
        classifier_choice = "Mô hình phân loại (ResNet50)"
        if app_mode == "Phát hiện vật thể + Phân loại":
            classifier_choice = st.selectbox(
                "Chọn mô hình phân loại",
                options=[
                    "Mô hình phân loại (ResNet50)"
                ],
                index=0,
                help="Chọn mô hình phân loại rác cho vùng vật thể phát hiện."
            )
            
        # Defaults
        score_threshold = 0.45
        nms_iou_threshold = 0.40
        cls_override_threshold = 0.60
        
        # Show detection sliders if not in classification-only mode
        if app_mode != "Phân loại ảnh đơn (ResNet50)":
            score_threshold = st.slider(
                "Ngưỡng điểm phát hiện (Score Thresh)", 
                min_value=0.01, max_value=0.99, value=0.45, step=0.01,
                help="Ngưỡng tin cậy tối thiểu để Faster R-CNN giữ lại bounding box."
            )
            nms_iou_threshold = st.slider(
                "Ngưỡng trùng lặp NMS (IoU Thresh)", 
                min_value=0.10, max_value=0.95, value=0.40, step=0.01,
                help="Ngưỡng triệt tiêu các hộp trùng lặp (Non-Maximum Suppression)."
            )
            
            if app_mode == "Phát hiện vật thể + Phân loại" and classifier_choice == "Mô hình phân loại (ResNet50)":
                cls_override_threshold = st.slider(
                    "Ngưỡng ghi đè (ResNet50 Override)", 
                    min_value=0.30, max_value=0.99, value=0.60, step=0.01,
                    help="Nếu ResNet50 phân loại crop đạt độ tin cậy lớn hơn mức này, nó sẽ ghi đè nhãn của Faster R-CNN."
                )

        st.divider()

        # Session Statistics calculations
        total_items = len(history)
        avg_c = np.mean([h["conf"] for h in history]) if history else 0
        high_c = np.max([h["conf"] for h in history]) if history else 0

        st.markdown("<div class='section-title'>Thống kê phiên</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{total_items}</div>
                <div class='metric-label'>Đã quét</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_c:.0f}%</div>
                <div class='metric-label'>TB tin cậy</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            effective_classes = [h.get("corrected") or h["label"] for h in history]
            most = Counter(effective_classes).most_common(1)
            most_label = CLASS_INFO[most[0][0]]["icon"] + " " + most[0][0] if most else "—"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:0.85rem;padding-top:0.3rem'>{most_label}</div>
                <div class='metric-label'>Nhiều nhất</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{high_c:.0f}%</div>
                <div class='metric-label'>Cao nhất</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Live breakdown of classes
        st.markdown("<div class='section-title'>Danh mục rác đã quét</div>", unsafe_allow_html=True)
        for name, info in CLASS_INFO.items():
            cnt = sum(1 for h in history if (h.get("corrected") or h["label"]) == name)
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.6rem;
                        padding:0.35rem 0.2rem;border-bottom:1px solid #2E3E2E;font-size:0.83rem;'>
                <span>{info['icon']}</span>
                <span style='flex:1;color:#A8C8A8'>{info['vi']}</span>
                <span style='font-family:"IBM Plex Mono",monospace;
                             color:{"#6ACA6A" if cnt > 0 else "#3E5040"};font-size:0.78rem;'>{cnt}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("🗑  Xóa lịch sử"):
            st.session_state.history = []
            st.session_state.last_processed_key = None
            st.rerun()
            
    return app_mode, detector_path, classifier_choice, score_threshold, nms_iou_threshold, cls_override_threshold
