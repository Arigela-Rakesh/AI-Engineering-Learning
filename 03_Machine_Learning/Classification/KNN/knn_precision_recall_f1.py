import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

df = pd.DataFrame(
    iris.data,
)

columns = iris.feature_names

df["Species"] = iris.target

X = df.drop("Species", axis=1)

Y = df["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=52
)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Actual:", Y_test.values)

print("Predicted", y_pred)

precision = precision_score(Y_test, y_pred, average="weighted")

recall = recall_score(Y_test, y_pred, average="weighted")

f1_score = f1_score(Y_test, y_pred, average="weighted")

cm = confusion_matrix(Y_test, y_pred)

print(precision)

print(recall)

print(f1_score)

print(cm)
