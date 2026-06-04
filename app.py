# To create a machine learning project that predicts whether a student will pass or fail based on study hours, attendance, and previous scores. The project must be managed using GitHub with proper version coontrol and documentation.# Step 1: Set up the GitHub repository
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.title("Student Pass/Fail Prediction")

# Load Dataset
data = pd.read_csv("student_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Data Preprocessing
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col2:
    st.metric("F1 Score", f"{f1:.2f}")

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(ax=ax)

st.pyplot(fig)

# Prediction Section
st.subheader("Predict Student Result")

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

    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    probability = model.predict_proba(
        [[study_hours, attendance, previous_score]]
    )[0]

    if prediction == 1:
        st.success("Student is likely to PASS")
    else:
        st.error("Student is likely to FAIL")

    st.write(
        f"Pass Probability: {probability[1]*100:.2f}%"
    )                  