import pickle
from pathlib import Path

import numpy as np
import streamlit as st


class Sav:
    """Utility wrapper for pickle-based model files (.sav/.pkl)."""

    @staticmethod
    def load(path):
        """Load a serialized object from a .sav or .pkl file."""
        file_path = Path(path)
        candidates = [file_path]

        if file_path.suffix.lower() not in {".sav", ".pkl"}:
            candidates.extend([
                file_path.with_suffix(".sav"),
                file_path.with_suffix(".pkl"),
            ])

        for candidate in candidates:
            if candidate.exists():
                with candidate.open("rb") as file:
                    return pickle.load(file)

        raise FileNotFoundError(f"Model file not found: {file_path}")

    @staticmethod
    def save(path, obj):
        """Save an object to a .sav file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as file:
            pickle.dump(obj, file)
        return file_path


MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.sav"
loaded_model = Sav.load(MODEL_PATH)

# 1. Custom CSS for Visual Styling (Colors, Fonts, Backgrounds)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5dc;
        }

        h1 {
            color: #5c4033;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
        }

        p {
            font-size: 18px;
            color: #333333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Diabetes Risk Predictor")
st.write("---")

st.markdown(
    "<p style='text-align: center;'>Enter your health metrics below for an instant preliminary screening.</p>",
    unsafe_allow_html=True,
)
st.write("")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age:", min_value=0, max_value=120, value=0, step=1)
    glucose = st.number_input("Glucose Level (mg/dL):", min_value=0, value=0, step=1)
    insulin = st.number_input("Insulin Level (µU/mL):", min_value=0, value=0, step=1)
    diabetespedigreefunction = st.number_input("Diabetes pedigree function:", min_value=0.0, value=0.0, step=0.01)
    blood_pressure = st.number_input("Blood Pressure (mm Hg):", min_value=0, value=0, step=1)

with col2:
    weight = st.number_input("Weight (kg):", min_value=1.0, value=70.0, step=0.5)
    height_cm = st.number_input("Height (cm):", min_value=1.0, value=170.0, step=0.5)
    number_of_pregnancies = st.number_input("Number of pregnancies:", min_value=0, value=0, step=1)
    skinthickness = st.number_input("Skin thickness (mm):", min_value=0, value=0, step=1)
    st.write("---")

if st.button("Predict Diabetes Risk", use_container_width=True):
    if height_cm <= 0 or weight <= 0:
        st.warning("Please enter valid height and weight values greater than 0.")
    else:
        height_m = height_cm / 100
        calculated_bmi = weight / (height_m ** 2)
        st.info(f"Your calculated BMI is: {calculated_bmi:.1f}")

        input_data = np.array(
            [[
                number_of_pregnancies,
                glucose,
                blood_pressure,
                skinthickness,
                insulin,
                calculated_bmi,
                diabetespedigreefunction,
                age,
            ]],
            dtype=float,
        )

        prediction = loaded_model.predict(input_data)

        st.write("---")
        if prediction[0] == 1:
            st.error("⚠️ HIGH RISK: The model indicates a high probability of diabetes. Please consult a doctor.")
        else:
            st.success("✅ LOW RISK: The model indicates a low probability of diabetes. Keep up the healthy lifestyle!")
else:
    st.info("Please fill in the valid details to get a prediction.")


