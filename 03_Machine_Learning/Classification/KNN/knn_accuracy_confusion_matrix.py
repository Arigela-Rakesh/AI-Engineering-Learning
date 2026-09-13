import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["Species"] = iris.target

X = df.drop("Species", axis=1)
Y = df["Species"]


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Actual", Y_test.values)
print("Predicted", y_pred)

accuracy = accuracy_score(y_pred, Y_test)

print(accuracy)

cm = confusion_matrix(y_pred, Y_test)

print(cm)
