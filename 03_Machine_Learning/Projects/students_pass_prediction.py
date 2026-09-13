import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "PreviousMarks": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    "Pass": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
}

df = pd.DataFrame(data)

X = df[["StudyHours", "Attendance", "PreviousMarks"]]

y = df["Pass"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])

parameters = {"model__C": [0.1, 1, 10, 100]}

grid = GridSearchCV(pipeline, parameters, cv=5, scoring="f1")

grid.fit(X_train, y_train)

print("Best Parameters")

print(grid.best_params_)

y_pred = grid.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))
