import pandas as pd

Students = {
    "Name": ["Rakesh", "Ravi", "Raju", "Ramu"],
    "Age": [23, 24, 22, 26],
    "Marks": [89, 83, 95, 82],
    "City": ["Hyderabad", "Chennai", "Bangalore", "Delhi"],
}

df = pd.DataFrame(Students)
print(df)
print(df[df["Marks"] > 88])
print(df[df["Age"] >= 23])
print(df[(df["Marks"] > 85) & (df["Age"] > 22)])
print(df.sort_values("Marks"))
print(df.sort_values("Marks", ascending=False))
print(df.head(3))
print(df.tail(2))
print(df.sample(2))

departments = {"Department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "IT"]}
df1 = pd.DataFrame(departments)
print(df1["Department"].value_counts())

Employees = {
    "Name": ["A", "B", "C", "D", "E"],
    "Age": [25, 30, 28, 35, 27],
    "Salary": [45000, 75000, 60000, 90000, 55000],
    "Department": ["IT", "HR", "IT", "Finance", "IT"],
}

df2 = pd.DataFrame(Employees)
print(df2["Salary"] > 60000)
print(df2[(df2["Age"] > 27) & (df2["Salary"] > 50000)])
print(df2.sort_values("Salary", ascending=False))
print(df2.head(2))
print(df2["Department"].value_counts())
print(df2[df2["Department"] == "IT"])
