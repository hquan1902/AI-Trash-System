import streamlit as st
import pandas as pd
from datetime import datetime
from Demo.utils import CLASS_INFO

def render_history_tab(history):
    """
    Renders the 'Lịch sử quét' (History logs) tab.
    Allows CSV logs export and custom styled HTML logs list.
    """
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    
    if not history:
        st.info("Chưa có lịch sử quét rác.")
        return
        
    # Build Pandas DataFrame for CSV download
    df_rows = []
    for i, h in enumerate(history):
        effective = h.get("corrected") or h["label"]
        fb_text = "Chưa phản hồi"
        if h.get("is_correct") is True:
            fb_text = "Đúng"
        elif h.get("is_correct") is False:
            fb_text = f"Sai (Sửa -> {effective})"
            
        df_rows.append({
            "STT": i + 1,
            "Chế độ": "Phân loại" if h["type"] == "classification" else "Nhận diện",
            "Nhãn gốc (AI)": h["label"],
            "Nhãn hiệu chỉnh": effective,
            "Phản hồi": fb_text,
            "Tên Việt": CLASS_INFO[effective]["vi"],
            "Độ tin cậy (%)": f"{h['conf']:.2f}",
            "Ngày": h.get("date", ""),
            "Giờ": h["time"],
        })
        
    df = pd.DataFrame(df_rows)
    
    # Download Button (CSV)
    st.download_button(
        label="⬇️  Xuất lịch sử (CSV)",
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"ecosort_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="history_csv_download"
    )
    
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Toàn bộ lịch sử dự đoán</div>", unsafe_allow_html=True)
    
    # Custom CSS Table UI
    st.markdown("""
    <div style='display:flex;gap:0.6rem;padding:0.4rem 0.8rem;
                background:#EEF4EE;border-radius:8px 8px 0 0;
                font-size:0.72rem;text-transform:uppercase;
                letter-spacing:0.08em;color:#5A7A5A;font-weight:600;'>
        <span style='min-width:2rem'>#</span>
        <span style='min-width:4.5rem'>Chế độ</span>
        <span style='flex:1'>Nhãn phân tích</span>
        <span style='min-width:4rem;text-align:center'>Phản hồi</span>
        <span style='min-width:3.5rem;text-align:right'>Tin cậy</span>
        <span style='min-width:9rem;text-align:right'>Thời gian</span>
    </div>
    <div style='background:#FFFFFF;border:1px solid #DDE8DD;
                border-top:none;border-radius:0 0 10px 10px;overflow:hidden;'>
    """, unsafe_allow_html=True)
    
    for i, h in enumerate(reversed(history)):
        effective = h.get("corrected") or h["label"]
        info = CLASS_INFO[effective]
        bg = "#FAFCFA" if i % 2 == 0 else "#FFFFFF"
        
        # Mode text
        mode_badge = "Phân loại" if h["type"] == "classification" else "Nhận diện"
        
        # Feedback badge
        is_correct = h.get("is_correct")
        if is_correct is True:
            fb_badge = "<span style='font-size:0.72rem;background:#E6F4EA;color:#2A6A2A;padding:2px 7px;border-radius:99px;'>✅ Đúng</span>"
        elif is_correct is False and h.get("corrected"):
            fb_badge = f"<span style='font-size:0.72rem;background:#FFF3CD;color:#856404;padding:2px 7px;border-radius:99px;'>✏️ Sửa</span>"
        elif is_correct is False:
            fb_badge = "<span style='font-size:0.72rem;background:#FCE8E8;color:#8A2A2A;padding:2px 7px;border-radius:99px;'>❌ Sai</span>"
        else:
            fb_badge = "<span style='font-size:0.72rem;color:#9AAA9A;'>—</span>"
            
        orig_label = h["label"]
        if effective != orig_label:
            label_html = (f"{effective} "
                          f"<span style='color:#AAAAAA;font-size:0.73rem;text-decoration:line-through'>{orig_label}</span>"
                          f" <span style='font-size:0.75rem;color:#7A9A7A;font-weight:400;margin-left:0.2rem'>{info['vi']}</span>")
        else:
            label_html = (f"{effective} "
                          f"<span style='font-size:0.75rem;color:#7A9A7A;font-weight:400;margin-left:0.4rem'>{info['vi']}</span>")
                          
        st.markdown(f"""
        <div class='hist-row' style='background:{bg}'>
            <span style='min-width:2rem;font-size:0.75rem;color:#9AAA9A;
                         font-family:"IBM Plex Mono",monospace;'>
                {len(history) - i}
            </span>
            <span style='min-width:4.5rem;font-size:0.75rem;color:#666;'>
                {mode_badge}
            </span>
            <span style='font-size:1.1rem;min-width:1.6rem'>{info['icon']}</span>
            <span class='hist-label'>{label_html}</span>
            <span style='min-width:4rem;text-align:center'>{fb_badge}</span>
            <span class='hist-conf'>{h['conf']:.1f}%</span>
            <span class='hist-time'>{h.get('date','')} {h['time']}</span>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
