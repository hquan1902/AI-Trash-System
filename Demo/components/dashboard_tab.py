import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from Demo.utils import CLASS_INFO, USER_CLASS_NAMES

def render_dashboard_tab(history):
    """
    Renders the 'Dashboard' tab showing detailed statistics and graphs.
    """
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    
    if not history:
        st.info("Chưa có dữ liệu thống kê. Vui lòng thực hiện phân loại hoặc quét ít nhất 1 ảnh.")
        return
        
    # Dashboard Overview Metrics
    total_items = len(history)
    avg_c = np.mean([h["conf"] for h in history])
    max_c = np.max([h["conf"] for h in history])
    
    effective_classes = [h.get("corrected") or h["label"] for h in history]
    unique_types = len(set(effective_classes))
    
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in zip(
        [m1, m2, m3, m4],
        [total_items, f"{avg_c:.1f}%", f"{max_c:.1f}%", unique_types],
        ["Tổng vật thể đã quét", "Độ tin cậy TB", "Độ tin cậy cao nhất", "Các loại rác xuất hiện"],
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)
        
    # Feedback analysis
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Thống kê phản hồi người dùng</div>", unsafe_allow_html=True)
    
    feedback_hist = [h for h in history if h.get("is_correct") is not None]
    n_feedback = len(feedback_hist)
    n_correct = sum(1 for h in feedback_hist if h.get("is_correct") is True)
    n_wrong = sum(1 for h in feedback_hist if h.get("is_correct") is False)
    accuracy_rate = (n_correct / n_feedback * 100) if n_feedback > 0 else None
    
    fb1, fb2, fb3, fb4 = st.columns(4)
    acc_val = f"{accuracy_rate:.1f}%" if accuracy_rate is not None else "—"
    acc_color = ("#2A6A2A" if accuracy_rate and accuracy_rate >= 70
                 else "#B05A00" if accuracy_rate else "#7A9A7A")
    
    for col, val, lbl, color in zip(
        [fb1, fb2, fb3, fb4],
        [n_feedback, n_correct, n_wrong, acc_val],
        ["Đã phản hồi", "✅ Đúng (AI)", "❌ Sai (đã sửa)", "Độ chính xác xác nhận"],
        ["#1A3A6A", "#2A6A2A", "#8A2A2A", acc_color],
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color};font-size:1.7rem'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    
    # Grid layout for charts
    d1, d2 = st.columns(2, gap="large")
    
    with d1:
        st.markdown("<div class='section-title'>Phân bố loại rác (Nhãn xác nhận)</div>", unsafe_allow_html=True)
        # Pie Chart
        counter = Counter(effective_classes)
        labels_pie = [c for c in USER_CLASS_NAMES if counter.get(c, 0) > 0]
        
        if labels_pie:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            fig1.patch.set_facecolor("#FFFFFF")
            sizes_pie = [counter[l] for l in labels_pie]
            colors_pie = [CLASS_INFO[l]["color"] for l in labels_pie]
            icons_pie = [CLASS_INFO[l]["icon"] + " " + l for l in labels_pie]
            
            wedges, texts, autotexts = ax1.pie(
                sizes_pie, labels=icons_pie, autopct="%1.0f%%",
                colors=colors_pie, startangle=90, pctdistance=0.75,
                wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 2},
            )
            for t in texts:
                t.set_fontsize(8); t.set_color("#2A4A2A")
            for at in autotexts:
                at.set_fontsize(8); at.set_color("#1A3A1A"); at.set_fontweight("500")
            ax1.set_facecolor("#FFFFFF")
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
        else:
            st.info("Chưa đủ dữ liệu vẽ phân bố hình tròn.")
            
    with d2:
        st.markdown("<div class='section-title'>Số lần xuất hiện từng loại rác</div>", unsafe_allow_html=True)
        # Bar Chart
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#FFFFFF")
        ax2.set_facecolor("#FAFCFA")
        
        counter = Counter(effective_classes)
        sorted_labels = sorted(USER_CLASS_NAMES, key=lambda c: counter.get(c, 0), reverse=True)
        counts = [counter.get(c, 0) for c in sorted_labels]
        bar_cols = [CLASS_INFO[c]["color"] for c in sorted_labels]
        
        y_pos = np.arange(len(sorted_labels))
        ax2.barh(y_pos, counts, color=bar_cols, edgecolor="#FFFFFF", linewidth=1, height=0.55)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{CLASS_INFO[c]['icon']} {c}" for c in sorted_labels], fontsize=8.5, color="#2A4A2A")
        ax2.invert_yaxis()
        ax2.set_xlabel("Số lần phát hiện", fontsize=8.5, color="#7A9A7A")
        ax2.tick_params(axis="x", labelsize=8, colors="#9AAA9A")
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.grid(axis="x", color="#EEF4EE", linewidth=0.8)
        for sp in ["top", "right", "bottom"]:
            ax2.spines[sp].set_visible(False)
        ax2.spines["left"].set_color("#DDE8DD")
        for i, v in enumerate(counts):
            if v > 0:
                ax2.text(v + 0.05, i, str(v), va="center", fontsize=8.5, color="#2A6A2A", fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
    # Error / Success rate per class
    if n_feedback > 0:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Độ chính xác theo từng loại rác (Dựa trên phản hồi)</div>", unsafe_allow_html=True)
        
        class_correct = {c: 0 for c in USER_CLASS_NAMES}
        class_wrong = {c: 0 for c in USER_CLASS_NAMES}
        for h in feedback_hist:
            pred = h["label"]
            if h["is_correct"] is True:
                class_correct[pred] += 1
            else:
                class_wrong[pred] += 1
                
        fig4, ax4 = plt.subplots(figsize=(10, 3.2))
        fig4.patch.set_facecolor("#FFFFFF")
        ax4.set_facecolor("#FAFCFA")
        x4 = np.arange(len(USER_CLASS_NAMES))
        w = 0.38
        
        bars_ok = ax4.bar(x4 - w/2, [class_correct[c] for c in USER_CLASS_NAMES], width=w, color="#5BC85B", label="✅ Đúng", edgecolor="#FFFFFF")
        bars_err = ax4.bar(x4 + w/2, [class_wrong[c] for c in USER_CLASS_NAMES], width=w, color="#E87070", label="❌ Sai", edgecolor="#FFFFFF")
        
        ax4.set_xticks(x4)
        ax4.set_xticklabels([f"{CLASS_INFO[c]['icon']} {c}" for c in USER_CLASS_NAMES], fontsize=8.5, color="#3A5A3A")
        ax4.set_ylabel("Số lần", fontsize=8.5, color="#7A9A7A")
        ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax4.tick_params(axis="y", labelsize=8, colors="#9AAA9A")
        ax4.grid(axis="y", color="#EEF4EE", linewidth=0.8, zorder=0)
        ax4.legend(fontsize=8.5, framealpha=0)
        for sp in ["top", "right"]:
            ax4.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax4.spines[sp].set_color("#DDE8DD")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig4)
        plt.close()
        
    # Confidence score history line graph
    if total_items >= 2:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Độ tin cậy qua các lần quét</div>", unsafe_allow_html=True)
        
        fig3, ax3 = plt.subplots(figsize=(10, 2.5))
        fig3.patch.set_facecolor("#FFFFFF")
        ax3.set_facecolor("#FAFCFA")
        
        conf_vals = [h["conf"] for h in history]
        x_vals = np.arange(1, len(conf_vals) + 1)
        
        ax3.fill_between(x_vals, conf_vals, alpha=0.12, color="#3A9A3A")
        ax3.plot(x_vals, conf_vals, color="#3A9A3A", linewidth=1.5, marker="o", markersize=4, markerfacecolor="#FFFFFF", markeredgecolor="#3A9A3A", markeredgewidth=1.5)
        ax3.axhline(y=avg_c, color="#9ABB9A", linestyle="--", linewidth=1, label=f"Trung bình: {avg_c:.1f}%")
        
        ax3.set_xlim(0.5, len(conf_vals) + 0.5)
        ax3.set_ylim(0, 105)
        ax3.set_xlabel("Vật thể thứ", fontsize=8.5, color="#7A9A7A")
        ax3.set_ylabel("Độ tin cậy (%)", fontsize=8.5, color="#7A9A7A")
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax3.tick_params(labelsize=8, colors="#9AAA9A")
        ax3.grid(color="#EEF4EE", linewidth=0.6)
        ax3.legend(fontsize=8, framealpha=0)
        for sp in ["top", "right"]:
            ax3.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax3.spines[sp].set_color("#DDE8DD")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3)
        plt.close()
