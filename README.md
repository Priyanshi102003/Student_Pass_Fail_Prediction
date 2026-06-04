# Student Pass/Fail Prediction

## Overview

This project predicts whether a student will pass or fail based on:

* Study Hours
* Attendance Percentage
* Previous Scores

The model is trained using a Random Forest Classifier.

## Dataset

Features:

1. Study_Hours
2. Attendance
3. Previous_Score

Target:

* Pass (1)
* Fail (0)

## Preprocessing

* Loaded dataset using Pandas
* Checked for missing values
* Split data into training and testing sets

## Model

Random Forest Classifier

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Train model

python src/train_model.py

3. App.py
   streamlit run app.py


