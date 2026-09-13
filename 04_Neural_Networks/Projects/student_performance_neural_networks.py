import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

n_samples = 500

study_hours = np.random.uniform(1, 10, n_samples)

attendance = np.random.uniform(50, 100, n_samples)

previous_marks = np.random.uniform(30, 100, n_samples)

score = study_hours * 5 + attendance * 0.3 + previous_marks * 0.5

passed = (score > 75).astype(int)

df = pd.DataFrame(
    {
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousMarks": previous_marks,
        "Pass": passed,
    }
)

print(df.head())

X = df[["StudyHours", "Attendance", "PreviousMarks"]]

y = df["Pass"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=500,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

y_val_pred = model.predict(X_val_scaled)

y_test_pred = model.predict(X_test_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", val_accuracy)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Validation Accuracy:", val_accuracy)

print(classification_report(y_test, y_test_pred))

print(confusion_matrix(y_test, y_test_pred))
