"""Local web interface for the diabetic retinopathy research model.

Run with: streamlit run retinopathy_web_app.py
Educational/research use only. This is not a clinical diagnostic device.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Change this only if the trained V2 model is stored somewhere else.
MODEL_PATH = Path(__file__).resolve().parent / "retinopathy_output_v2" / "final_retinopathy_model.keras"
IMAGE_SIZE = 300
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_COLORS = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
CLASS_EMOJIS = ["✅", "🟡", "🟠", "🔴", "🚨"]
CLASS_DESCRIPTIONS = [
    "No signs of diabetic retinopathy detected.",
    "Mild non-proliferative diabetic retinopathy — microaneurysms only.",
    "Moderate NPDR — more than just microaneurysms but less than severe.",
    "Severe NPDR — extensive intraretinal hemorrhages, venous beading.",
    "Proliferative DR — neovascularization and/or vitreous/preretinal hemorrhage.",
]

# ──────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaXAI · DR Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0f1117;
    --bg-card: rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.07);
    --border-glass: rgba(255, 255, 255, 0.08);
    --text-primary: #f0f2f6;
    --text-secondary: #9ca3af;
    --accent-cyan: #06b6d4;
    --accent-purple: #8b5cf6;
    --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
    --shadow-glow: 0 0 40px rgba(6, 182, 212, 0.15);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1b2e 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Hero header gradient text ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.03em;
    animation: fadeSlideIn 0.8s ease-out;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
    font-weight: 400;
    animation: fadeSlideIn 1s ease-out;
}

/* ── Glass cards ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 0.75rem 0;
    box-shadow: var(--shadow-glow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
    box-shadow: 0 0 60px rgba(6, 182, 212, 0.2);
}

/* ── Result card (colored left border) ── */
.result-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 0.75rem 0;
    box-shadow: var(--shadow-glow);
    animation: fadeSlideIn 0.6s ease-out;
}

/* ── Severity badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    animation: pulseGlow 2s ease-in-out infinite;
}

/* ── Metric big number ── */
.big-metric {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-top: 0.25rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(6, 182, 212, 0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(6, 182, 212, 0.6) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(6, 182, 212, 0.45) !important;
}

/* ── Progress bars ── */
.prob-bar-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0;
}
.prob-bar-label {
    min-width: 120px;
    font-weight: 500;
    font-size: 0.9rem;
}
.prob-bar-track {
    flex: 1;
    height: 10px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.prob-bar-value {
    min-width: 50px;
    text-align: right;
    font-weight: 600;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
}

/* ── Animated divider ── */
.gradient-divider {
    height: 2px;
    background: var(--accent-gradient);
    border: none;
    border-radius: 2px;
    margin: 1.5rem 0;
    opacity: 0.5;
}

/* ── Info cards in sidebar ── */
.info-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.info-pill-icon {
    font-size: 1.1rem;
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(6,182,212,0.2); }
    50%      { box-shadow: 0 0 20px rgba(6,182,212,0.4); }
}

/* ── Hide default Streamlit chrome ── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* ── Streamlit alerts restyle ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border-glass) !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Image container ── */
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border-glass);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model loading & preprocessing (unchanged)
# ──────────────────────────────────────────────
@st.cache_resource
def load_saved_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the V2 training preprocessing: RGB, crop black border, resize."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    brightness = pixels.max(axis=2)
    rows = np.where(np.any(brightness > 10, axis=1))[0]
    columns = np.where(np.any(brightness > 10, axis=0))[0]
    if len(rows) and len(columns):
        pixels = pixels[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    pixels = tf.image.resize(pixels, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    return np.expand_dims(tf.cast(pixels, tf.float32).numpy(), axis=0)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size:2.5rem;">👁️</span>
        <h2 style="margin:0.3rem 0 0; font-weight:700; font-size:1.4rem;
                    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            RetinaXAI
        </h2>
        <p style="color:#9ca3af; font-size:0.8rem; margin-top:0.2rem;">
            AI-Powered DR Screening
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Model Details")
    st.markdown("""
    <div class="info-pill"><span class="info-pill-icon">🏗️</span> Architecture: EfficientNetB0</div>
    <div class="info-pill"><span class="info-pill-icon">📐</span> Input: 300 × 300 RGB</div>
    <div class="info-pill"><span class="info-pill-icon">🎯</span> Classes: 5 (DR severity)</div>
    <div class="info-pill"><span class="info-pill-icon">📊</span> Output: Softmax probabilities</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📋 DR Severity Scale")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(
            f'<div class="info-pill">'
            f'<span class="info-pill-icon">{CLASS_EMOJIS[i]}</span>'
            f'<span style="color:{CLASS_COLORS[i]}; font-weight:600;">{i}</span>'
            f'&nbsp;— {name}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2);
                border-radius:10px; padding:0.8rem; margin-top:0.5rem;">
        <p style="margin:0; font-size:0.78rem; color:#fca5a5;">
            ⚠️ <strong>Disclaimer</strong><br>
            This tool is an educational research prototype.<br>
            It is <strong>not</strong> a certified medical device and must
            <strong>not</strong> guide clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main area — Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <div class="hero-title">RetinaXAI</div>
    <div class="hero-subtitle">
        AI-Powered Diabetic Retinopathy Screening &nbsp;·&nbsp;
        EfficientNetB0 &nbsp;·&nbsp; Research Prototype
    </div>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Upload section
# ──────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="text-align:center;">
    <h3 style="margin:0 0 0.3rem; font-weight:600;">
        📤 &nbsp;Upload Retinal Fundus Image
    </h3>
    <p style="color:#9ca3af; font-size:0.9rem; margin:0;">
        Drag and drop or browse for a JPG, JPEG, or PNG fundus photograph
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload one retinal fundus image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        # Two-column layout: image + action
        col_img, col_action = st.columns([3, 2], gap="large")
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(uploaded_image, caption="Uploaded retinal image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_action:
            st.markdown("""
            <div class="glass-card">
                <h4 style="margin:0 0 0.5rem;">🔍 &nbsp;Ready to Analyze</h4>
                <p style="color:#9ca3af; font-size:0.88rem; margin:0 0 1rem;">
                    The model will classify this image into one of five
                    diabetic retinopathy severity grades. Click below to run inference.
                </p>
            </div>
            """, unsafe_allow_html=True)

            analyze_clicked = st.button("⚡  Analyze Image", type="primary", use_container_width=True)

        # ── Run inference ──
        if analyze_clicked:
            with st.spinner("Loading model and analyzing image…"):
                model = load_saved_model()
                scores = model.predict(preprocess(uploaded_image), verbose=0)[0]

            predicted_grade = int(np.argmax(scores))
            confidence = float(scores[predicted_grade]) * 100
            color = CLASS_COLORS[predicted_grade]
            emoji = CLASS_EMOJIS[predicted_grade]
            label = CLASS_NAMES[predicted_grade]

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            # ── Result header ──
            res_col1, res_col2, res_col3 = st.columns([2, 2, 3], gap="large")

            with res_col1:
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">Prediction</div>
                    <div class="big-metric" style="color:{color};">
                        Grade {predicted_grade}
                    </div>
                    <div class="severity-badge" style="background:{color}22; color:{color}; margin-top:0.75rem;">
                        {emoji} {label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">Confidence</div>
                    <div class="big-metric" style="color:{color};">
                        {confidence:.1f}<span style="font-size:1.5rem;">%</span>
                    </div>
                    <p style="color:#9ca3af; font-size:0.78rem; margin-top:0.5rem;">
                        Softmax probability for the predicted class
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">Clinical Description</div>
                    <p style="margin-top:0.5rem; font-size:0.95rem; line-height:1.6;">
                        {CLASS_DESCRIPTIONS[predicted_grade]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown("""
            <div class="glass-card" style="margin-top:1rem;">
                <h4 style="margin:0 0 1rem; font-weight:600;">
                    📊 &nbsp;Class Probabilities
                </h4>
            """, unsafe_allow_html=True)

            for i, (name, score) in enumerate(zip(CLASS_NAMES, scores)):
                pct = float(score) * 100
                is_predicted = i == predicted_grade
                bar_opacity = "1" if is_predicted else "0.55"
                st.markdown(f"""
                <div class="prob-bar-container" style="opacity:{bar_opacity};">
                    <span class="prob-bar-label" style="color:{CLASS_COLORS[i]};">
                        {CLASS_EMOJIS[i]} {name}
                    </span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill"
                             style="width:{pct}%; background:{CLASS_COLORS[i]};"></div>
                    </div>
                    <span class="prob-bar-value" style="color:{CLASS_COLORS[i]};">
                        {pct:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <p style="text-align:center; color:#6b7280; font-size:0.78rem;
                      margin-top:1rem; font-style:italic;">
                Confidence reflects the model's softmax output — it is not a measure of
                medical or clinical certainty.
            </p>
            """, unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Could not analyze the image: {error}")
else:
    # ── Empty state ──
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:3rem 2rem;">
        <span style="font-size:3.5rem; display:block; margin-bottom:1rem;">🔬</span>
        <h3 style="font-weight:600; margin:0 0 0.5rem;">
            No Image Uploaded Yet
        </h3>
        <p style="color:#9ca3af; max-width:480px; margin:0 auto; font-size:0.95rem;">
            Upload a retinal fundus photograph above to begin AI-powered
            diabetic retinopathy screening. The model will classify the image
            into one of five severity grades.
        </p>
    </div>
    """, unsafe_allow_html=True)
