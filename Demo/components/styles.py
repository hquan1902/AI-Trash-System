import streamlit as st

def inject_custom_css():
    """Inject custom CSS styles for the ECOSORT AI unified web app."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #F4F6F4; color: #1A2A1A; }

[data-testid="stSidebar"] {
    background: #1A2A1A !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #C8D8C0 !important; }
[data-testid="stSidebar"] hr { border-color: #2E3E2E !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2E3E2E !important;
    color: #C8D8C0 !important;
    border: 1px solid #3E5040 !important;
    border-radius: 8px !important;
    width: 100%;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #DDE8DD;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #2A6A2A;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #7A9A7A;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #DDE8DD;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.result-class {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #1A4A1A;
}
.result-vi { font-size: 0.95rem; color: #5A8A5A; margin-top: 0.2rem; }
.conf-track {
    background: #EEF4EE;
    border-radius: 99px;
    height: 8px;
    margin-top: 0.8rem;
}
.conf-fill {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #3A9A3A, #5BC85B);
}
.tip-box {
    background: #F0F8F0;
    border-left: 3px solid #4A8A4A;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #4A7A4A;
}

.crop-card {
    background: #FFFFFF;
    border: 1px solid #EAEAEA;
    border-radius: 12px;
    padding: 0.9rem;
    margin-bottom: 0.8rem;
    display: flex;
    gap: 1rem;
    align-items: center;
}
.crop-img-container {
    width: 80px;
    height: 80px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #DDE8DD;
    background: #FAFCFA;
    display: flex;
    justify-content: center;
    align-items: center;
}
.crop-details {
    flex: 1;
}
.crop-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1A3A1A;
}
.crop-subtitle {
    font-size: 0.82rem;
    color: #6A8A6A;
}
.crop-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 99px;
    font-weight: 500;
    margin-left: 0.4rem;
    display: inline-block;
}
.badge-agree { background: #E6F4EA; color: #1E7E34; }
.badge-override { background: #FFF3CD; color: #856404; }
.badge-keep { background: #E8F0FE; color: #1A73E8; }
.badge-det { background: #F1F3F4; color: #5F6368; }

.top3-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0.9rem;
    background: #FAFCFA;
    border: 1px solid #E4EEE4;
    border-radius: 10px;
    margin-bottom: 0.45rem;
}
.top3-name  { flex: 1; font-size: 0.88rem; color: #2A4A2A; }
.top3-conf  { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; color: #3A8A3A; font-weight: 500; }

.hist-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #EEF4EE;
    font-size: 0.83rem;
}
.hist-row:last-child { border-bottom: none; }
.hist-label { flex: 1; color: #1A3A1A; font-weight: 500; }
.hist-conf  { font-family: 'IBM Plex Mono', monospace; color: #3A8A3A; min-width: 3.5rem; text-align: right; }
.hist-time  { color: #8AAA8A; min-width: 5rem; text-align: right; }

.section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7A9A7A;
    font-weight: 600;
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #DDE8DD;
}

[data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #DDE8DD !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] { border-radius: 7px !important; font-size: 0.88rem !important; }
[aria-selected="true"] { background: #1A2A1A !important; color: #FFFFFF !important; }
            
[data-baseweb="tab"] {
    color: #1A2A1A !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: #1A2A1A !important;
    color: #FFFFFF !important;
}
[data-baseweb="tab"]:hover {
    background: #EEF4EE !important;
    color: #1A2A1A !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
