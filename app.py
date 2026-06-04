import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("student_model.pkl")

st.title("Student Pass/Fail Prediction")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=60
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("Student is likely to PASS")
    else:
        st.error("Student is likely to FAIL")

    st.write(
        f"Pass Probability: {probability[1] * 100:.2f}%"
    )