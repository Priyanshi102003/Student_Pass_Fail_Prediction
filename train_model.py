import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset inside code
data = pd.DataFrame({
    "Study_Hours": [2,3,4,5,6,7,8,9,10,1,12,3,6,4,8],
    "Attendance": [60,65,70,75,80,85,90,92,95,50,98,55,78,68,88],
    "Previous_Score": [45,50,55,60,65,70,75,80,85,40,90,48,67,52,76],
    "Pass": [0,0,0,1,1,1,1,1,1,0,1,0,1,0,1]
})

X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

joblib.dump(model, "student_model.pkl")
print("Model saved as student_model.pkl")