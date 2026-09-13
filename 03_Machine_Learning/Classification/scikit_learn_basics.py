import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.feature_names)

print(iris.target_names)

print(iris.data)

print(iris.target)

print(iris.data.shape)

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.head())

df["Species"] = iris.target

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df["Species"].value_counts())

X = df.drop("Species", axis=1)

Y = df["Species"]

print(X.head())

print(Y.head())

print(X.shape)

print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
