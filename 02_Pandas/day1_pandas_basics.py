import pandas as pd

# program1-Series
series = pd.Series([10, 20, 30, 40, 50])
print(series)

# Program2-DataFrame
students = {
    "Name": ["Rakesh", "Rahul", "Anitha"],
    "Age": [23, 22, 21],
    "Marks": [85, 90, 95],
}
df = pd.DataFrame(students)
print(df)
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
df.to_csv("students1.csv", index=False)

print(pd.read_csv("students1.csv"))

# Mini Challenge
Employees = {
    "Employess_ID": [101, 102, 103],
    "Name": ["Rakesh", "Naveen", "Shiva"],
    "Department": ["IT", "HR", "Finance"],
    "Salary": [95000, 75000, 45000],
}
df1 = pd.DataFrame(Employees)
print(df1)
print(df1.shape)
print(df1.columns)
print(df1.dtypes)
