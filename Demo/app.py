import streamlit as st
import torch
from Demo.utils import load_user_classifier, load_friend_detector

# Components Imports
from Demo.components.styles import inject_custom_css
from Demo.components.sidebar import render_sidebar
from Demo.components.scan_tab import render_scan_tab
from Demo.components.dashboard_tab import render_dashboard_tab
from Demo.components.history_tab import render_history_tab

# 1. Page Configuration
st.set_page_config(
    page_title="EcoSort AI — Unified Demo",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Styles
inject_custom_css()

# 3. Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_key" not in st.session_state:
    st.session_state.last_processed_key = None
if "active_ids" not in st.session_state:
    st.session_state.active_ids = []

# 5. Render Sidebar (Run first to determine which models to load)
sidebar_out = render_sidebar(st.session_state.history)
app_mode = sidebar_out[0]
detector_path = sidebar_out[1]
classifier_choice = sidebar_out[2]
score_threshold = sidebar_out[3]
nms_iou_threshold = sidebar_out[4]
cls_override_threshold = sidebar_out[5]

# 4. Lazy Load Models (Cached)
@st.cache_resource(show_spinner=False)
def get_cached_classifier():
    return load_user_classifier()

@st.cache_resource(show_spinner=False)
def get_cached_detector(model_path):
    return load_friend_detector(model_path)

# Load detector model dynamically based on selection
detector_model = None
device_det = "N/A"
if app_mode != "🔍 Phân loại ảnh đơn (ResNet50)":
    try:
        detector_model = get_cached_detector(detector_path)
        device_det = "GPU" if next(detector_model.parameters()).is_cuda else "CPU"
    except Exception as e:
        import os
        st.error(f"❌ Không load được mô hình Faster R-CNN ({os.path.basename(detector_path)}): {e}")
        st.stop()

# Load classifier model if selected
classifier_model = None
device_cls = "N/A"
if app_mode == "🔍 Phân loại ảnh đơn (ResNet50)" or (app_mode == "🤝 Phát hiện vật thể + Phân loại" and classifier_choice == "Mô hình của tôi (ResNet50)"):
    try:
        classifier_model = get_cached_classifier()
        device_cls = "GPU" if next(classifier_model.parameters()).is_cuda else "CPU"
    except Exception as e:
        st.error(f"❌ Không load được mô hình ResNet50 (Phân loại): {e}")
        st.stop()

# 6. Main App Header
st.markdown("""
<div style='padding:1.8rem 0 0.6rem;'>
    <span style='font-family:"IBM Plex Mono",monospace;font-size:1.8rem;
                 font-weight:500;color:#1A3A1A;letter-spacing:-0.5px;'>
        Waste Classification & Detection
    </span><br>
    <span style='font-size:0.88rem;color:#7A9A7A;'>
        Tải ảnh hoặc chụp camera — kết hợp sức mạnh phân loại của ResNet50 và nhận diện vật thể Faster R-CNN
    </span>
</div>
""", unsafe_allow_html=True)

col_s = st.columns([1.2, 1.2, 1.2, 3.5])
with col_s[0]:
    st.markdown(f"""
    <div style='background:#EAFAEA;border:1px solid #B8D8B8;border-radius:6px;
                padding:0.3rem 0.7rem;font-size:0.78rem;color:#3A7A3A;
                text-align:center;font-family:"IBM Plex Mono",monospace;'>
        ✓ ResNet50: {device_cls}
    </div>""", unsafe_allow_html=True)
with col_s[1]:
    st.markdown(f"""
    <div style='background:#EAFAEA;border:1px solid #B8D8B8;border-radius:6px;
                padding:0.3rem 0.7rem;font-size:0.78rem;color:#3A7A3A;
                text-align:center;font-family:"IBM Plex Mono",monospace;'>
        ✓ Faster R-CNN: {device_det}
    </div>""", unsafe_allow_html=True)
with col_s[2]:
    st.markdown(f"""
    <div style='background:#EAFAEA;border:1px solid #B8D8B8;border-radius:6px;
                padding:0.3rem 0.7rem;font-size:0.78rem;color:#3A7A3A;
                text-align:center;font-family:"IBM Plex Mono",monospace;'>
        ✓ 7 Lớp Rác Chuẩn
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#DDE8DD;margin:1.2rem 0'>", unsafe_allow_html=True)

# 7. Setup Main Layout Tabs
tab_scan, tab_dash, tab_hist = st.tabs(["🔍  Phân loại & Phát hiện", "📊  Dashboard", "📋  Lịch sử quét"])

with tab_scan:
    render_scan_tab(
        classifier_model=classifier_model,
        detector_model=detector_model,
        app_mode=app_mode,
        detector_path=detector_path,
        classifier_choice=classifier_choice,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        cls_override_threshold=cls_override_threshold
    )

with tab_dash:
    render_dashboard_tab(st.session_state.history)

with tab_hist:
    render_history_tab(st.session_state.history)
