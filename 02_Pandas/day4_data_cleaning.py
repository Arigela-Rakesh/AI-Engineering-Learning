import pandas as pd
import numpy as np

employees = {
    "Name": ["A", "B", None, "D"],
    "Age": [23, None, 25, 26],
    "Salary": [50000, 60000, None, 70000],
}
df = pd.DataFrame(employees)
print(df.isnull())
print(df.isnull().sum())
df["Name"] = df["Name"].fillna("UnKnown")
df["Age"] = df["Age"].fillna(0)
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
print(df)
print(df.dropna())
print(df.dropna(axis=1))

students = {"Name": ["A", "B", "B", "C", "C"], "Marks": [90, 80, 85, 95, 95]}

df1 = pd.DataFrame(students)
print(df1.drop_duplicates())

df.rename(columns={"Name": "Employee Name", "Salary": "Monthly Salary"})

df["Age"] = df["Age"].astype(int)

Employees1 = {
    "Name": ["A", "B", "C", "D", np.nan],
    "Age": [25, np.nan, 28, 35, 30],
    "Salary": [45000, 65000, np.nan, 90000, 55000],
    "Department": ["IT", "HR", "IT", "Finance", "IT"],
}

df2 = pd.DataFrame(Employees1)
print(df2.isnull().sum())
df2["Name"] = df2["Name"].fillna("UnKnown")
df2["Age"] = df2["Age"].fillna(df2["Age"].mean())
df2["Salary"] = df2["Salary"].fillna(df2["Salary"].mean())
df2.rename(
    columns={
        "Salary": "Monthly Salary",
    }
)
df2["Age"] = df2["Age"].astype(int)

print(df2)
