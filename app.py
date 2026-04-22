import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI | Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK MEDICAL TERMINAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    /* Base */
    .stApp { background-color: #020b18; color: #c9d1d9; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1200px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020b18 0%, #0a1628 100%);
        border-right: 1px solid #00ff9520;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Header */
    .mediscan-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #00ff9530;
        margin-bottom: 2rem;
    }
    .mediscan-title {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff95, #00b4ff, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 4px;
        margin: 0;
    }
    .mediscan-subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #00ff9580;
        font-size: 0.85rem;
        letter-spacing: 3px;
        margin-top: 0.5rem;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1f2d, #0a1628);
        border: 1px solid #00ff9520;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: border-color 0.3s;
    }
    .metric-card:hover { border-color: #00ff9560; }

    /* Disease tabs */
    .disease-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .disease-desc {
        font-family: 'Rajdhani', sans-serif;
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Sliders & inputs */
    .stSlider > div > div > div { background: #00ff9530 !important; }
    .stSlider > div > div > div > div { background: #00ff95 !important; }
    div[data-testid="stNumberInput"] input {
        background: #0d1f2d !important;
        border: 1px solid #00ff9540 !important;
        color: #00ff95 !important;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #00ff95, #00b4ff) !important;
        color: #020b18 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 20px #00ff9560 !important;
    }

    /* Result boxes */
    .result-safe {
        background: linear-gradient(135deg, #0d2818, #0a1f12);
        border: 2px solid #00ff95;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-danger {
        background: linear-gradient(135deg, #2d0d0d, #1f0a0a);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 900;
        margin: 0;
    }
    .result-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }

    /* Tips */
    .tip-box {
        background: #0d1f2d;
        border-left: 3px solid #00b4ff;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.95rem;
        color: #c9d1d9;
    }

    /* Section labels */
    .section-label {
        font-family: 'Share Tech Mono', monospace;
        color: #00ff9580;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #00ff9520;
        padding-bottom: 0.3rem;
    }

    /* Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0a1628;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.8rem !important;
        color: #8b949e !important;
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ff9520, #00b4ff20) !important;
        color: #00ff95 !important;
        border-bottom: 2px solid #00ff95 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "diabetes_model": joblib.load("models/diabetes_model.pkl"),
        "diabetes_scaler": joblib.load("models/diabetes_scaler.pkl"),
        "heart_model": joblib.load("models/heart_model.pkl"),
        "heart_scaler": joblib.load("models/heart_scaler.pkl"),
        "parkinsons_model": joblib.load("models/parkinsons_model.pkl"),
        "parkinsons_scaler": joblib.load("models/parkinsons_scaler.pkl"),
    }

models = load_models()

# ─────────────────────────────────────────────
# HELPER: RISK GAUGE
# ─────────────────────────────────────────────
def risk_gauge(probability, title):
    color = "#ff4444" if probability > 0.5 else "#00ff95"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        title={"text": title, "font": {"color": "#c9d1d9", "family": "Orbitron", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "family": "Orbitron", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color},
            "bgcolor": "#0d1f2d",
            "bordercolor": "#00ff9530",
            "steps": [
                {"range": [0, 30], "color": "#0d2818"},
                {"range": [30, 60], "color": "#1f1a0a"},
                {"range": [60, 100], "color": "#2d0d0d"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={"color": "#c9d1d9"}
    )
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="mediscan-header">
    <p class="mediscan-title">🏥 MEDISCAN AI</p>
    <p class="mediscan-subtitle">[ MULTI-DISEASE EARLY RISK DETECTION SYSTEM v1.0 ]</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <p style='font-family: Orbitron; font-size: 1rem; color: #00ff95; letter-spacing: 2px;'>⚕ SYSTEM STATUS</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("🟢 **Diabetes Model** — Online")
    st.markdown("🟢 **Heart Disease Model** — Online")
    st.markdown("🟢 **Parkinson's Model** — Online")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='font-family: Share Tech Mono; color: #00ff9580; font-size: 0.75rem; letter-spacing: 1px;'>
    ⚠ DISCLAIMER<br><br>
    This tool is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='font-family: Share Tech Mono; color: #8b949e; font-size: 0.7rem;'>
    Built by Rishit Pandya<br>
    M.Data Science @ UniAdelaide<br>
    Powered by scikit-learn + Streamlit
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩸  DIABETES", "🫀  HEART DISEASE", "🫁  PARKINSON'S"])

# ══════════════════════════════════════════════
# TAB 1 — DIABETES
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="disease-header" style="color:#00ff95;">🩸 Diabetes Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="disease-desc">Enter patient vitals below. The AI model will predict diabetes risk based on the Pima Indians Diabetes Dataset.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-label">📊 Basic Information</p>', unsafe_allow_html=True)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, key="d_preg")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, key="d_age")
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 110, key="d_glucose")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70, key="d_bp")

    with col2:
        st.markdown('<p class="section-label">🔬 Clinical Measurements</p>', unsafe_allow_html=True)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, key="d_skin")
        insulin = st.slider("Insulin Level (μU/mL)", 0, 900, 80, key="d_insulin")
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1, key="d_bmi")
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, key="d_dpf")

    if st.button("🔍 ANALYSE DIABETES RISK", key="btn_diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age]])
        scaled = models["diabetes_scaler"].transform(input_data)
        prediction = models["diabetes_model"].predict(scaled)[0]
        probability = models["diabetes_model"].predict_proba(scaled)[0][1]

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-danger">
                    <p class="result-title" style="color:#ff4444;">⚠ HIGH RISK</p>
                    <p class="result-subtitle">Diabetes risk detected. Please consult a healthcare professional.</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🥗 Reduce sugar and refined carbohydrate intake",
                    "🏃 Aim for 30 minutes of exercise daily",
                    "💧 Stay well hydrated — drink 8+ glasses of water",
                    "🩺 Schedule a fasting blood glucose test immediately",
                    "⚖️ Work towards a healthy BMI (18.5–24.9)"
                ]
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <p class="result-title" style="color:#00ff95;">✅ LOW RISK</p>
                    <p class="result-subtitle">No significant diabetes risk detected. Keep up healthy habits!</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🥦 Maintain a balanced diet rich in vegetables",
                    "🏋️ Keep up regular physical activity",
                    "💤 Ensure 7–8 hours of quality sleep",
                    "🩺 Annual check-ups are still recommended",
                    "🚭 Avoid smoking and limit alcohol consumption"
                ]

        with col_gauge:
            st.plotly_chart(risk_gauge(probability, "DIABETES RISK SCORE"), use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top:1rem;">💡 HEALTH RECOMMENDATIONS</p>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — HEART DISEASE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="disease-header" style="color:#ff6b6b;">🫀 Heart Disease Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="disease-desc">Enter cardiovascular indicators below. Model trained on the Cleveland Heart Disease Dataset.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-label">📊 Patient Profile</p>', unsafe_allow_html=True)
        h_age = st.number_input("Age", min_value=1, max_value=120, value=45, key="h_age")
        h_sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="h_sex")
        h_cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x], key="h_cp")
        h_trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, key="h_bp")
        h_chol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200, key="h_chol")
        h_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes", key="h_fbs")
        h_restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                  format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x], key="h_ecg")

    with col2:
        st.markdown('<p class="section-label">🔬 Stress Test & Advanced</p>', unsafe_allow_html=True)
        h_thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150, key="h_thal")
        h_exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes", key="h_exang")
        h_oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1, key="h_oldpeak")
        h_slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], key="h_slope")
        h_ca = st.selectbox("Major Vessels Coloured by Fluoroscopy", options=[0, 1, 2, 3, 4], key="h_ca")
        h_thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                               format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x], key="h_thal2")

    if st.button("🔍 ANALYSE HEART DISEASE RISK", key="btn_heart"):
        input_data = np.array([[h_age, h_sex, h_cp, h_trestbps, h_chol, h_fbs,
                                  h_restecg, h_thalach, h_exang, h_oldpeak, h_slope, h_ca, h_thal]])
        scaled = models["heart_scaler"].transform(input_data)
        prediction = models["heart_model"].predict(scaled)[0]
        probability = models["heart_model"].predict_proba(scaled)[0][1]

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown("""
                <div class="result-danger">
                    <p class="result-title" style="color:#ff4444;">⚠ HIGH RISK</p>
                    <p class="result-subtitle">Cardiovascular risk detected. Seek medical attention promptly.</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🩺 Consult a cardiologist as soon as possible",
                    "🧂 Reduce sodium intake to lower blood pressure",
                    "🚭 Stop smoking — it doubles heart disease risk",
                    "🏃 Begin a medically supervised exercise program",
                    "💊 Discuss cholesterol medication with your doctor"
                ]
            else:
                st.markdown("""
                <div class="result-safe">
                    <p class="result-title" style="color:#00ff95;">✅ LOW RISK</p>
                    <p class="result-subtitle">No significant cardiovascular risk detected. Keep your heart healthy!</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🥑 Eat heart-healthy foods — avocado, nuts, fish",
                    "🏊 Cardio exercise 3–5 times per week",
                    "😌 Manage stress through meditation or yoga",
                    "🩺 Check blood pressure and cholesterol annually",
                    "🍷 Limit alcohol to recommended guidelines"
                ]

        with col_gauge:
            st.plotly_chart(risk_gauge(probability, "HEART RISK SCORE"), use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top:1rem;">💡 HEALTH RECOMMENDATIONS</p>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — PARKINSON'S
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="disease-header" style="color:#00b4ff;">🫁 Parkinson\'s Disease Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="disease-desc">Enter voice measurement biomarkers below. The model uses vocal frequency analysis to detect Parkinson\'s risk.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-label">🎙️ Vocal Frequency Measures</p>', unsafe_allow_html=True)
        fo = st.slider("MDVP:Fo(Hz) — Avg Vocal Frequency", 80.0, 270.0, 150.0, step=0.1, key="p_fo")
        fhi = st.slider("MDVP:Fhi(Hz) — Max Vocal Frequency", 100.0, 600.0, 200.0, step=0.1, key="p_fhi")
        flo = st.slider("MDVP:Flo(Hz) — Min Vocal Frequency", 60.0, 240.0, 100.0, step=0.1, key="p_flo")
        jitter_pct = st.slider("MDVP:Jitter(%)", 0.0, 1.0, 0.005, step=0.001, key="p_jitter")
        jitter_abs = st.slider("MDVP:Jitter(Abs)", 0.0, 0.0001, 0.00003, step=0.000001, format="%.6f", key="p_jabs")
        rap = st.slider("MDVP:RAP", 0.0, 0.02, 0.003, step=0.0001, key="p_rap")
        ppq = st.slider("MDVP:PPQ", 0.0, 0.02, 0.003, step=0.0001, key="p_ppq")
        ddp = st.slider("Jitter:DDP", 0.0, 0.06, 0.009, step=0.001, key="p_ddp")
        shimmer = st.slider("MDVP:Shimmer", 0.0, 0.2, 0.03, step=0.001, key="p_shimmer")
        shimmer_db = st.slider("MDVP:Shimmer(dB)", 0.0, 2.0, 0.3, step=0.01, key="p_shimdb")
        apq3 = st.slider("Shimmer:APQ3", 0.0, 0.1, 0.015, step=0.001, key="p_apq3")

    with col2:
        st.markdown('<p class="section-label">📈 Nonlinear Dynamics Measures</p>', unsafe_allow_html=True)
        apq5 = st.slider("Shimmer:APQ5", 0.0, 0.15, 0.02, step=0.001, key="p_apq5")
        apq = st.slider("MDVP:APQ", 0.0, 0.15, 0.025, step=0.001, key="p_apq")
        dda = st.slider("Shimmer:DDA", 0.0, 0.3, 0.045, step=0.001, key="p_dda")
        nhr = st.slider("NHR — Noise-to-Harmonics Ratio", 0.0, 0.35, 0.025, step=0.001, key="p_nhr")
        hnr = st.slider("HNR — Harmonics-to-Noise Ratio", 5.0, 40.0, 22.0, step=0.1, key="p_hnr")
        rpde = st.slider("RPDE", 0.2, 0.7, 0.45, step=0.001, key="p_rpde")
        dfa = st.slider("DFA", 0.5, 0.9, 0.72, step=0.001, key="p_dfa")
        spread1 = st.slider("Spread1", -8.0, -2.0, -5.5, step=0.01, key="p_spread1")
        spread2 = st.slider("Spread2", 0.0, 0.5, 0.2, step=0.001, key="p_spread2")
        d2 = st.slider("D2", 1.5, 4.0, 2.3, step=0.01, key="p_d2")
        ppe = st.slider("PPE", 0.0, 0.5, 0.2, step=0.001, key="p_ppe")

    if st.button("🔍 ANALYSE PARKINSON'S RISK", key="btn_park"):
        input_data = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp,
                                  shimmer, shimmer_db, apq3, apq5, apq, dda,
                                  nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        scaled = models["parkinsons_scaler"].transform(input_data)
        prediction = models["parkinsons_model"].predict(scaled)[0]
        probability = models["parkinsons_model"].predict_proba(scaled)[0][1]

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown("""
                <div class="result-danger">
                    <p class="result-title" style="color:#ff4444;">⚠ HIGH RISK</p>
                    <p class="result-subtitle">Parkinson's indicators detected in vocal analysis. Please consult a neurologist.</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🧠 Consult a neurologist for proper diagnosis",
                    "🏋️ Exercise — studies show it slows progression",
                    "🎙️ Speech therapy can help manage vocal symptoms",
                    "💊 Discuss Levodopa therapy options with your doctor",
                    "🤝 Join a Parkinson's support group for guidance"
                ]
            else:
                st.markdown("""
                <div class="result-safe">
                    <p class="result-title" style="color:#00ff95;">✅ LOW RISK</p>
                    <p class="result-subtitle">No significant Parkinson's indicators detected. Great news!</p>
                </div>
                """, unsafe_allow_html=True)
                tips = [
                    "🧠 Keep your brain active — puzzles, reading, learning",
                    "🏃 Regular aerobic exercise protects neurological health",
                    "🥗 Mediterranean diet supports brain health",
                    "😴 Prioritise quality sleep for neural repair",
                    "🩺 Annual neurological check-ups after age 50"
                ]

        with col_gauge:
            st.plotly_chart(risk_gauge(probability, "PARKINSON'S RISK SCORE"), use_container_width=True)

        st.markdown('<p class="section-label" style="margin-top:1rem;">💡 HEALTH RECOMMENDATIONS</p>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)