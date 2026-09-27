"""
Heart Disease Prediction App — Streamlit
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
)

# ---------- Custom HTML/CSS ----------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    h1, h2, h3, p, label, .stMarkdown { color: #f5f7fa !important; }
    .hero {
        text-align: center; padding: 2rem 1rem 1rem 1rem;
    }
    .hero h1 { font-size: 2.8rem; margin-bottom: 0.2rem; }
    .hero p { color: #b8c6d1 !important; font-size: 1.1rem; }
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }
    .result-ok {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px; padding: 2rem; text-align: center;
        color: white; font-size: 1.4rem; font-weight: 700;
    }
    .result-risk {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        border-radius: 16px; padding: 2rem; text-align: center;
        color: white; font-size: 1.4rem; font-weight: 700;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ef473a, #cb2d3e);
        color: white; border: none; border-radius: 12px;
        padding: 0.7rem 2rem; font-size: 1.1rem; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    # Train from CSV at startup — no saved model file needed
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv("heart_disease_health_data.csv").dropna()
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(df[features], df["target"])
    return {"model": model, "features": features}

bundle = load_model()
model, FEATURES = bundle["model"], bundle["features"]

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Predictor</h1>
        <p>Random Forest model trained on the UCI Heart Disease dataset (303 real patient records)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input form ----------
left, right = st.columns([2, 1])

with left:
    st.subheader("Enter patient details")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest pain type",
            [(0, "Typical angina"), (1, "Atypical angina"),
             (2, "Non-anginal pain"), (3, "Asymptomatic")],
            format_func=lambda x: x[1],
        )[0]
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG",
            [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "LV hypertrophy")],
            format_func=lambda x: x[1],
        )[0]
    with c2:
        thalach = st.slider("Max heart rate achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox(
            "Slope of peak exercise ST",
            [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
            format_func=lambda x: x[1],
        )[0]
        ca = st.selectbox("Major vessels colored by fluoroscopy (0–3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            [(1, "Normal"), (2, "Fixed defect"), (3, "Reversible defect")],
            format_func=lambda x: x[1],
        )[0]

    predict = st.button("🔍 Predict")

# ---------- Prediction ----------
with right:
    st.subheader("Prediction")

    if predict:
        row = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURES,
        )
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]

        if pred == 1:
            st.markdown(
                f'<div class="result-risk">⚠️ High risk of heart disease<br>'
                f'<span style="font-size:1rem">Confidence: {proba[1]*100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-ok">✅ Low risk — likely healthy<br>'
                f'<span style="font-size:1rem">Confidence: {proba[0]*100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

        st.progress(float(proba[1]), text=f"Risk probability: {proba[1]*100:.1f}%")
        st.caption("⚕️ Educational tool only — not medical advice.")
    else:
        st.info("Fill in the patient details and press **Predict**.")

# ---------- Feature importance ----------
st.subheader("What the model looks at")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(imp, horizontal=True)

st.markdown(
    "<p style='text-align:center;color:#8fa3b0'>Built with Streamlit · Random Forest · UCI Heart Disease dataset</p>",
    unsafe_allow_html=True,
)
