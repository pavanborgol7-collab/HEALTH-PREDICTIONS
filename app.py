import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "heart_disease_model.pkl"


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"⚠️ Could not find `{MODEL_PATH}`. Place the trained pipeline "
        "(preprocessor + RandomForestClassifier) in the same folder as this script."
    )
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction Dashboard")

st.markdown(
    """
    ### Machine Learning Prediction System
    Enter the patient's medical information below to generate
    a prediction using a **Random Forest classification model**.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("📋 About")

    st.info(
        "Enter the patient's clinical information carefully. "
        "The model will process the values through the same "
        "preprocessing pipeline used during training."
    )

    st.markdown("---")

    st.caption("Model")
    st.write("🌲 Random Forest Classifier")

    st.caption("Preprocessing")
    st.write("StandardScaler (numeric) + OneHotEncoder (categorical)")

    st.markdown("---")

    st.caption(
        "⚠️ This application is for educational purposes only "
        "and must **not** be used as a real medical diagnosis. "
        "Always consult a qualified physician."
    )

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
# NOTE: option codes below match the exact values the model was
# trained on (Cleveland heart-disease dataset conventions).

with st.form("patient_form"):

    st.subheader("👤 Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)

    with col2:
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
        )

    with col3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 – Typical angina",
                2: "2 – Atypical angina",
                3: "3 – Non-anginal pain",
                4: "4 – Asymptomatic",
            }[x],
            help="Type of chest pain experienced by the patient.",
        )

    st.subheader("🩺 Vital & Blood Information")
    col4, col5, col6 = st.columns(3)

    with col4:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50, max_value=250, value=120,
        )

    with col5:
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=50, max_value=700, value=200,
        )

    with col6:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    st.subheader("📈 Exercise Test Results")
    col7, col8, col9 = st.columns(3)

    with col7:
        thalach = st.number_input(
            "Max Heart Rate Achieved",
            min_value=50, max_value=250, value=150,
        )

    with col8:
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col9:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        )

    col10, col11, col12 = st.columns(3)

    with col10:
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – ST-T wave abnormality",
                2: "2 – Left ventricular hypertrophy",
            }[x],
        )

    with col11:
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 – Upsloping",
                2: "2 – Flat",
                3: "3 – Downsloping",
            }[x],
        )

    with col12:
        thal = st.selectbox(
            "Thalassemia",
            options=[3, 6, 7],
            format_func=lambda x: {
                3: "3 – Normal",
                6: "6 – Fixed defect",
                7: "7 – Reversible defect",
            }[x],
        )

    ca = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy (ca)",
        options=[0, 1, 2, 3],
    )

    st.markdown("")
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    with st.expander("🔎 View input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

    try:
        prediction = model.predict(input_df)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    st.subheader("🧾 Prediction Result")

    result_col, prob_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("### ⚠️ High risk of heart disease detected")
        else:
            st.success("### ✅ Low risk of heart disease detected")

    with prob_col:
        if proba is not None:
            # class order follows model.classes_
            classes = list(model.classes_)
            disease_idx = classes.index(1) if 1 in classes else 1
            disease_prob = proba[disease_idx]
            st.metric("Predicted probability of disease", f"{disease_prob * 100:.1f}%")
            st.progress(min(max(disease_prob, 0.0), 1.0))

    st.caption(
        "This result is generated by a machine learning model for educational "
        "purposes and is **not** a substitute for professional medical advice."
    )