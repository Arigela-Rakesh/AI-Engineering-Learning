import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

data = {
    "Experience": [1, 2, 3, 4, 5],
    "Age": [22, 25, 28, 30, 32],
    "Salary": [25000, 30000, 38000, 45000, 52000],
}

df = pd.DataFrame(data)

X = df[["Experience", "Age"]]

Y = df["Salary"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=52
)


model = LinearRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Actual", Y_test.values)

print("Predicted", y_pred)
