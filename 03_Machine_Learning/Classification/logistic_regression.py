import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

data = {"Hours": [1, 2, 3, 4, 5, 6], "Pass": [0, 0, 0, 1, 1, 1]}
df = pd.DataFrame(data)

X = df[["Hours"]]

Y = df["Pass"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Actual", Y_test.values)

print("Predicted", y_pred)
