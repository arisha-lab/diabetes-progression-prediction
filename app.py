
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_linear_model.joblib")
scaler = joblib.load("diabetes_scaler.joblib")

# Page configuration
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🧬",
    layout="centered"
)

# Title
st.title("🧬 Diabetes Progression Predictor")

st.write(
    "This application uses a Linear Regression machine learning model "
    "to predict a numerical diabetes progression score."
)

st.info(
    "Educational project only. This application is not a medical "
    "diagnostic tool and should not be used for clinical decision-making."
)

st.markdown("### About the inputs")

st.write(
    "The features below use the standardized numerical representation "
    "provided by the scikit-learn diabetes dataset. They are not direct "
    "clinical measurements such as ordinary BMI or blood pressure values."
)

st.markdown("### Enter standardized feature values")

# Input features
age = st.number_input(
    "Age",
    min_value=-0.2,
    max_value=0.2,
    value=0.0,
    step=0.01
)

sex = st.number_input(
    "Sex",
    min_value=-0.05,
    max_value=0.05,
    value=0.0,
    step=0.01
)

bmi = st.number_input(
    "BMI",
    min_value=-0.1,
    max_value=0.2,
    value=0.0,
    step=0.01
)

bp = st.number_input(
    "Blood Pressure (BP)",
    min_value=-0.12,
    max_value=0.13,
    value=0.0,
    step=0.01
)

s1 = st.number_input(
    "S1",
    min_value=-0.13,
    max_value=0.16,
    value=0.0,
    step=0.01
)

s2 = st.number_input(
    "S2",
    min_value=-0.12,
    max_value=0.20,
    value=0.0,
    step=0.01
)

s3 = st.number_input(
    "S3",
    min_value=-0.1,
    max_value=0.18,
    value=0.0,
    step=0.01
)

s4 = st.number_input(
    "S4",
    min_value=-0.1,
    max_value=0.19,
    value=0.0,
    step=0.01
)

s5 = st.number_input(
    "S5",
    min_value=-0.13,
    max_value=0.13,
    value=0.0,
    step=0.01
)

s6 = st.number_input(
    "S6",
    min_value=-0.14,
    max_value=0.14,
    value=0.0,
    step=0.01
)

# Prediction
if st.button("🔬 Predict Diabetes Progression Score"):

    input_data = pd.DataFrame(
        [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]],
        columns=[
            "age", "sex", "bmi", "bp",
            "s1", "s2", "s3", "s4", "s5", "s6"
        ]
    )

    # Apply the same preprocessing used during training
    input_scaled = scaler.transform(input_data)

    # Generate prediction
    prediction = model.predict(input_scaled)[0]

    st.success(
        f"### Predicted Diabetes Progression Score: {prediction:.2f}"
    )

    st.caption(
        "Model: Linear Regression | Test MAE: 42.79 | Test R²: 0.45"
    )
