import streamlit as st
import hashlib
import io
from PIL import Image

# Modular component sub-imports
from Demo.components.scan_inference import run_inference_if_new_state
from Demo.components.scan_left_panel import render_left_panel
from Demo.components.scan_right_panel import render_right_panel

def render_scan_tab(
    classifier_model, 
    detector_model, 
    app_mode, 
    score_threshold, 
    nms_iou_threshold, 
    cls_override_threshold
):
    """
    Main orchestration entrypoint for rendering the 'Phân loại & Phát hiện' tab.
    Divided into modular panels (left preview and right analytics).
    """
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    
    inp_upload, inp_camera = st.tabs(["📁 Upload ảnh", "📷 Camera"])
    image_input = None
    
    with inp_upload:
        uploaded = st.file_uploader(
            "Chọn ảnh",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            key="scan_file_uploader"
        )
        if uploaded:
            image_input = Image.open(uploaded).convert("RGB")
            
    with inp_camera:
        cam = st.camera_input("Chụp ảnh", label_visibility="collapsed", key="scan_camera_input")
        if cam:
            image_input = Image.open(cam).convert("RGB")
            
    if image_input:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        left_col, right_col = st.columns([1.3, 1.0], gap="large")
        
        # Calculate Image MD5 Key
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        img_key = hashlib.md5(buf.getvalue()).hexdigest()
        
        # Check current run configuration state key
        current_state_key = f"{img_key}_{app_mode}_{score_threshold}_{nms_iou_threshold}_{cls_override_threshold}"
        
        # 1. Run model inferences
        run_inference_if_new_state(
            classifier_model=classifier_model,
            detector_model=detector_model,
            image_input=image_input,
            img_key=img_key,
            current_state_key=current_state_key,
            app_mode=app_mode,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            cls_override_threshold=cls_override_threshold
        )
        
        # Filter active entries for display
        current_entries = [h for h in st.session_state.history if h["id"] in st.session_state.active_ids]
        
        # 2. Render Left Column (Visual predictions overlay)
        with left_col:
            render_left_panel(
                image_input=image_input,
                app_mode=app_mode,
                current_entries=current_entries,
                cls_override_threshold=cls_override_threshold
            )
            
        # 3. Render Right Column (Classification card or crops list with feedbacks)
        with right_col:
            render_right_panel(
                image_input=image_input,
                app_mode=app_mode,
                current_entries=current_entries
            )
                
    else:
        st.markdown("""
        <div style='text-align:center;padding:4.5rem 1rem;background:#FFFFFF;
                    border:1.5px dashed #C8D8C8;border-radius:14px;margin-top:0.5rem;'>
            <div style='font-size:2.8rem;margin-bottom:0.8rem'>🌿</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:1.05rem;
                        color:#2A6A2A;font-weight:500;'>HỆ THỐNG SẴN SÀNG QUÉT</div>
            <div style='color:#8AAA8A;font-size:0.85rem;margin-top:0.4rem;'>
                Vui lòng tải ảnh lên hoặc chụp từ camera để tiến hành phân tích
            </div>
        </div>
        """, unsafe_allow_html=True)
