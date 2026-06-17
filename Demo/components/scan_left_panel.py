import streamlit as st
import numpy as np
import torch

# Imports from Demo.utils
from Demo.utils import (
    draw_static_predictions,
    draw_fused_predictions,
    build_interactive_bbox_figure,
    CLASS_INFO,
    USER_CLASS_NAMES,
    DET_CLASS_NAMES,
    PLOTLY_AVAILABLE
)

def render_left_panel(image_input, app_mode, classifier_choice, current_entries, cls_override_threshold):
    """Renders the left panel containing the image preview and bounding boxes."""
    st.markdown("<div class='section-title'>Ảnh Đầu Vào & Nhận Diện</div>", unsafe_allow_html=True)
    
    if app_mode == "Phân loại ảnh đơn (ResNet50)":
        st.image(image_input, use_container_width=True)
        st.markdown(f"<div style='font-size:0.75rem;color:#8AAA8A;margin-top:0.3rem'>{image_input.width} × {image_input.height} px</div>", unsafe_allow_html=True)
        return
    else:
        image_np = np.array(image_input)
        det_entries = [e for e in current_entries if e["type"] == "detection"]
    
    if PLOTLY_AVAILABLE and len(det_entries) > 0:
        plotly_rows = []
        for idx, entry in enumerate(det_entries):
            x1, y1, x2, y2 = entry["box"]
            f_class = entry["corrected"] if entry["corrected"] else entry["label"]
            info = CLASS_INFO[f_class]
            
            hover_html = (
                f"📦 Vật thể #{idx + 1}<br>"
                f"<b>Phân loại:</b> {info['icon']} {info['vi']} ({f_class})<br>"
                f"<b>Tin cậy:</b> {entry['conf']:.1f}%<br>"
            )
            if entry["fusion_source"] == "cls_override":
                hover_html += f"<i>(ResNet50 sửa nhãn gốc {entry['det_class']})</i>"
            elif entry["fusion_source"] == "agree":
                hover_html += "<i>(Đồng thuận giữa 2 model)</i>"
                
            plotly_rows.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "hover_text": hover_html,
                "color": info["color"]
            })
            
        fig = build_interactive_bbox_figure(image_np, plotly_rows)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("💡 Di chuột lên viền bounding box để xem chi tiết vật thể.")
    else:
        if len(det_entries) > 0:
            use_resnet = (
                app_mode == "Phát hiện vật thể + Phân loại" and 
                classifier_choice == "Mô hình phân loại (ResNet50)"
            )
            if use_resnet:
                cls_res_fmt = [{"class": e["cls_class"], "score": e["cls_score"]/100.0} for e in det_entries]
                pred_raw_fmt = {
                    "boxes": torch.tensor([e["box"] for e in det_entries]),
                    "scores": torch.tensor([e["det_score"]/100.0 for e in det_entries]),
                    "labels": torch.tensor([DET_CLASS_NAMES.index(e["det_class"]) + 1 if e["det_class"] in DET_CLASS_NAMES else 1 for e in det_entries], dtype=torch.int64)
                }
                drawn_np = draw_fused_predictions(
                    image_np=image_np,
                    prediction=pred_raw_fmt,
                    cls_results=cls_res_fmt,
                    cls_override_threshold=cls_override_threshold
                )
            else:
                pred_raw_fmt = {
                    "boxes": torch.tensor([e["box"] for e in det_entries]),
                    "scores": torch.tensor([e["conf"]/100.0 for e in det_entries]),
                    "labels": torch.tensor([DET_CLASS_NAMES.index(e["det_class"]) + 1 if e["det_class"] in DET_CLASS_NAMES else 1 for e in det_entries], dtype=torch.int64)
                }
                drawn_np = draw_static_predictions(image_np=image_np, prediction=pred_raw_fmt)
            st.image(drawn_np, use_container_width=True)
        else:
            st.image(image_input, use_container_width=True)
            st.warning("⚠️ Không phát hiện vật thể nào với ngưỡng tin cậy hiện tại.")
            
    st.markdown(f"<div style='font-size:0.75rem;color:#8AAA8A;margin-top:0.3rem'>{image_input.width} × {image_input.height} px</div>", unsafe_allow_html=True)
