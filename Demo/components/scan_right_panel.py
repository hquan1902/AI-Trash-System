import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
from collections import Counter

# Imports from Demo.utils
from Demo.utils import (
    CLASS_INFO,
    RECYCLE_TIP,
    USER_CLASS_NAMES
)

from Demo.components.scan_inference import save_corrected_image

def render_classification_results(image_input, entry):
    """Renders classification results, feedback, and bar chart."""
    history_idx = st.session_state.history.index(entry)
    label = entry["label"]
    conf = entry["conf"]
    probs = np.array(entry["probs"])
    top3 = entry["top3"]
    
    display_label = entry["corrected"] if entry["corrected"] else label
    display_info = CLASS_INFO[display_label]
    
    corrected_badge = ""
    if entry["corrected"] and entry["corrected"] != label:
        corrected_badge = "<span class='crop-badge badge-override'>✏️ Đã sửa</span>"
        
    st.markdown(f"""
    <div class='result-card'>
        <div style='display:flex;align-items:flex-start;gap:1rem;'>
            <div style='font-size:3rem;line-height:1'>{display_info['icon']}</div>
            <div>
                <div class='result-class'>{display_label.upper()}{corrected_badge}</div>
                <div class='result-vi'>{display_info['vi']}</div>
            </div>
            <div style='margin-left:auto;text-align:right;'>
                <div style='font-family:"IBM Plex Mono",monospace;
                            font-size:1.6rem;font-weight:500;color:#2A6A2A;'>
                    {conf:.1f}%
                </div>
                <div style='font-size:0.72rem;color:#8AAA8A;
                            text-transform:uppercase;letter-spacing:0.06em;'>
                    TIN CẬY
                </div>
            </div>
        </div>
        <div class='conf-track'>
            <div class='conf-fill' style='width:{conf:.1f}%'></div>
        </div>
        <div class='tip-box'>💡 {RECYCLE_TIP[display_label]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # User feedback loop
    st.markdown("<div class='section-title'>Phản hồi người dùng</div>", unsafe_allow_html=True)
    if entry["is_correct"] is None:
        st.markdown("<div style='font-size:0.85rem;color:#3A5A3A;margin-bottom:0.5rem;'>Phân loại có <b>chính xác</b> không?</div>", unsafe_allow_html=True)
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("✅  Đúng rồi", key=f"fb_ok_{entry['id']}", use_container_width=True):
                st.session_state.history[history_idx]["is_correct"] = True
                st.session_state.history[history_idx]["corrected"] = label
                st.rerun()
        with fb_col2:
            if st.button("❌  Sai, muốn sửa", key=f"fb_err_{entry['id']}", use_container_width=True):
                st.session_state.history[history_idx]["is_correct"] = False
                st.rerun()
                
    elif entry["is_correct"] is True:
        st.markdown("<div style='background:#F0FFF0;border:1px solid #B8D8B8;border-radius:10px;padding:0.7rem 1rem;font-size:0.85rem;color:#2A6A2A;'>✅ Cảm ơn bạn đã xác nhận kết quả chính xác!</div>", unsafe_allow_html=True)
        
    elif entry["is_correct"] is False:
        if entry["corrected"] is None:
            st.markdown("<div style='font-size:0.85rem;color:#7A5A2A;margin-bottom:0.5rem;'>Hãy chọn loại rác đúng:</div>", unsafe_allow_html=True)
            cls_cols = st.columns(3)
            for ci, cls_name in enumerate(USER_CLASS_NAMES):
                c_info = CLASS_INFO[cls_name]
                btn_label = f"{c_info['icon']} {c_info['vi']}"
                if cls_name == label:
                    btn_label += " (AI)"
                with cls_cols[ci % 3]:
                    if st.button(btn_label, key=f"corr_{entry['id']}_{cls_name}", use_container_width=True):
                        st.session_state.history[history_idx]["corrected"] = cls_name
                        save_corrected_image(image_input, entry, cls_name)
                        st.rerun()
        else:
            correct_label = entry["corrected"]
            st.markdown(f"""
            <div style='background:#FFF8F0;border:1px solid #F0C8A0;border-radius:10px;
                        padding:0.7rem 1rem;font-size:0.85rem;color:#7A5A2A;'>
                ✏️ Đã sửa thành: <b>{CLASS_INFO[correct_label]['icon']} {CLASS_INFO[correct_label]['vi']}</b>
            </div>""", unsafe_allow_html=True)
            if st.button("↩ Thay đổi lại", key=f"recorr_{entry['id']}", use_container_width=False):
                st.session_state.history[history_idx]["corrected"] = None
                st.rerun()
                
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Top 3 dự đoán</div>", unsafe_allow_html=True)
    for idx, rank_idx in enumerate(top3):
        n = USER_CLASS_NAMES[rank_idx]
        medal = ["🥇", "🥈", "🥉"][idx]
        st.markdown(f"""
        <div class='top3-row'>
            <span style='font-size:1rem'>{medal}</span>
            <span style='font-size:1rem'>{CLASS_INFO[n]['icon']}</span>
            <span class='top3-name'>{CLASS_INFO[n]['vi']}</span>
            <span class='top3-conf'>{probs[rank_idx]*100:.2f}%</span>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Phân bố xác suất 7 lớp</div>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFCFA")
    colors_bar = [CLASS_INFO[c]["color"] if c == label else "#DDE8DD" for c in USER_CLASS_NAMES]
    bars = ax.bar(np.arange(len(USER_CLASS_NAMES)), probs * 100, color=colors_bar, edgecolor="#FFFFFF", linewidth=1.2, width=0.55, zorder=3)
    for bar, p in zip(bars, probs * 100):
        if p > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=8, color="#2A4A2A", fontfamily="monospace")
    ax.set_xticks(np.arange(len(USER_CLASS_NAMES)))
    ax.set_xticklabels([f"{CLASS_INFO[c]['icon']}" for c in USER_CLASS_NAMES], fontsize=10)
    ax.set_ylabel("Xác suất (%)", fontsize=8, color="#7A9A7A")
    ax.set_ylim(0, max(probs * 100) * 1.25 + 5)
    ax.tick_params(axis="y", labelsize=8, colors="#9AAA9A")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color="#EEF4EE", linewidth=0.8, zorder=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#DDE8DD")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()

def render_detection_results(image_input, det_entries):
    """Renders detection crop list, feedback loop widgets, and count charts."""
    num_detected = len(det_entries)
    st.write(f"- Vật thể phát hiện: **{num_detected}**")
    
    if num_detected > 0:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        
        # Highlight most prominent item
        best_entry = max(det_entries, key=lambda x: x["conf"])
        best_class = best_entry["corrected"] if best_entry["corrected"] else best_entry["label"]
        best_info = CLASS_INFO[best_class]
        st.success(f"Nổi bật nhất: **{best_info['icon']} {best_info['vi'].upper()}** ({best_entry['conf']:.1f}% tin cậy)")
        
        st.markdown("<div class='section-title'>Danh sách chi tiết vật thể</div>", unsafe_allow_html=True)
        
        for idx, entry in enumerate(det_entries):
            history_idx = st.session_state.history.index(entry)
            
            x1, y1, x2, y2 = [int(v) for v in entry["box"]]
            image_np = np.array(image_input)
            crop_np = image_np[y1:y2, x1:x2]
            
            f_class = entry["corrected"] if entry["corrected"] else entry["label"]
            info = CLASS_INFO[f_class]
            
            # Badge HTML markup
            badge_html = ""
            if entry["fusion_source"] == "agree":
                badge_html = "<span class='crop-badge badge-agree'>Đồng thuận 🤝</span>"
            elif entry["fusion_source"] == "cls_override":
                badge_html = "<span class='crop-badge badge-override'>ResNet50 sửa ✏️</span>"
            elif entry["fusion_source"] == "det_keep":
                badge_html = "<span class='crop-badge badge-keep'>Giữ nhãn gốc 🎯</span>"
            else:
                badge_html = "<span class='crop-badge badge-det'>Nhận diện 🎯</span>"
                
            if entry["corrected"]:
                badge_html += " <span class='crop-badge badge-override'>Đã sửa tay ✏️</span>"
                
            st.markdown(f"""
            <div class='crop-card'>
                <div class='crop-details'>
                    <div class='crop-title'>{info['icon']} {f_class.upper()} {badge_html}</div>
                    <div class='crop-subtitle'>{info['vi']} &nbsp;•&nbsp; Tin cậy: <b>{entry['conf']:.1f}%</b></div>
                    <div style='font-size:0.75rem;color:#999;margin-top:0.2rem;'>Toạ độ: [{x1}, {y1}, {x2}, {y2}]</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            crop_col1, crop_col2 = st.columns([1, 2.5])
            with crop_col1:
                if crop_np.size > 0:
                    crop_pil = Image.fromarray(crop_np)
                    st.image(crop_pil, use_container_width=True)
            
            with crop_col2:
                if entry["is_correct"] is None:
                    st.markdown("<span style='font-size:0.78rem;color:#4A6A4A;'>Nhãn chính xác?</span>", unsafe_allow_html=True)
                    fb_subcol1, fb_subcol2 = st.columns(2)
                    with fb_subcol1:
                        if st.button("✅ Đúng", key=f"ok_{entry['id']}", use_container_width=True):
                            st.session_state.history[history_idx]["is_correct"] = True
                            st.session_state.history[history_idx]["corrected"] = entry["label"]
                            st.rerun()
                    with fb_subcol2:
                        if st.button("❌ Sai", key=f"err_{entry['id']}", use_container_width=True):
                            st.session_state.history[history_idx]["is_correct"] = False
                            st.rerun()
                elif entry["is_correct"] is True:
                    st.markdown("<span style='font-size:0.78rem;color:#2A6A2A;font-weight:500;'>✓ Đã xác nhận chính xác</span>", unsafe_allow_html=True)
                else:
                    if entry["corrected"] is None:
                        corr_select = st.selectbox(
                            "Sửa thành nhãn đúng:",
                            options=["—"] + USER_CLASS_NAMES,
                            key=f"sel_{entry['id']}"
                        )
                        if corr_select != "—":
                            st.session_state.history[history_idx]["corrected"] = corr_select
                            save_corrected_image(image_input, entry, corr_select)
                            st.rerun()
                    else:
                        st.markdown(f"<span style='font-size:0.78rem;color:#856404;font-weight:500;'>✏️ Đã sửa: <b>{CLASS_INFO[entry['corrected']]['vi']}</b></span>", unsafe_allow_html=True)
                        if st.button("↩ Hoàn tác", key=f"undo_{entry['id']}"):
                            st.session_state.history[history_idx]["corrected"] = None
                            st.session_state.history[history_idx]["is_correct"] = None
                            st.rerun()
                            
            st.markdown("<div class='tip-box' style='margin-top:0.4rem;margin-bottom:1rem;'>💡 " + RECYCLE_TIP[f_class] + "</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:0.8rem 0;border-color:#EAEAEA;'>", unsafe_allow_html=True)
            
        st.markdown("<div class='section-title'>Phân bố loại vật thể trong ảnh</div>", unsafe_allow_html=True)
        counts = Counter([e["corrected"] or e["label"] for e in det_entries])
        
        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FAFCFA")
        
        classes_in_img = list(counts.keys())
        counts_in_img = list(counts.values())
        colors_bar = [CLASS_INFO[c]["color"] for c in classes_in_img]
        
        bars = ax.bar(np.arange(len(classes_in_img)), counts_in_img, color=colors_bar, width=0.45, zorder=3)
        ax.set_xticks(np.arange(len(classes_in_img)))
        ax.set_xticklabels([f"{CLASS_INFO[c]['icon']} {c}" for c in classes_in_img], fontsize=8, color="#2A4A2A")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(axis="y", labelsize=8, colors="#9AAA9A")
        ax.grid(axis="y", color="#EEF4EE", linewidth=0.8, zorder=0)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#DDE8DD")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Hãy thử giảm ngưỡng điểm phát hiện (Score Thresh) trong sidebar để tìm kiếm nhiều vật thể hơn.")

def render_right_panel(image_input, app_mode, current_entries):
    """Orchestrates left-right column routing for displaying detection/classification analytics."""
    st.markdown("<div class='section-title'>Kết Quả Phân Tích</div>", unsafe_allow_html=True)
    if app_mode == "🔍 Phân loại ảnh đơn (ResNet50)":
        if len(current_entries) > 0:
            render_classification_results(image_input, current_entries[0])
    else:
        det_entries = [e for e in current_entries if e["type"] == "detection"]
        render_detection_results(image_input, det_entries)
