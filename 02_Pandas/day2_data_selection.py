import pandas as pd

# Dataset
Employees = {
    "Name": ["Rakesh", "Rajesh", "Ravi", "Rahul"],
    "Age": [23, 25, 27, 26],
    "Salary": [95000, 85000, 75000, 76000],
}
df = pd.DataFrame(Employees)
print(df)

print(df["Name"])
print(df[["Age", "Salary"]])
print(df.loc[0])
print(df.iloc[0])
print(df.loc[1:3])
print(df.iloc[1:4])
print(df.loc[0:3, ["Name", "Salary"]])
print(df.iloc[0:4, [0, 2]])

Employees1 = {
    "Employee": ["A", "B", "C", "D"],
    "Experience": [2, 5, 3, 7],
    "Salary": [45000, 75000, 52000, 95000],
    "Rating": [4.2, 4.8, 4.5, 4.9],
}

df1 = pd.DataFrame(Employees1)
print(df1["Salary"])
print(df1[["Employee", "Rating"]])
print(df1.loc[0:1])
print(df1.loc[2:3])
print(df1.iloc[0:3, [0, 2]])
