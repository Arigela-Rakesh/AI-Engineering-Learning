import pandas as pd

employees = {
    "Name": ["A", "B", "C", "D", "E", "F"],
    "Department": ["IT", "HR", "IT", "Finance", "HR", "IT"],
    "Salary": [50000, 60000, 70000, 90000, 65000, 80000],
    "Experience": [2, 3, 5, 8, 4, 6],
}
df = pd.DataFrame(employees)
print(df)

print(df.groupby("Department")["Salary"].mean())
print(df.groupby("Department")["Salary"].sum())
print(df.groupby("Department")["Name"].count())
print(df.groupby("Department")["Salary"].max())
print(df.groupby("Department")["Salary"].min())
print(df.groupby("Department")["Salary"].agg(["mean", "max", "min", "count", "sum"]))

sales = {
    "Employee": ["A", "B", "A", "C", "B", "A", "C", "D"],
    "Product": [
        "Laptop",
        "Phone",
        "Phone",
        "Laptop",
        "Laptop",
        "Tablet",
        "Phone",
        "Tablet",
    ],
    "Sales": [50000, 30000, 25000, 60000, 45000, 20000, 35000, 40000],
}

df1 = pd.DataFrame(sales)
print(df1)

print(df1.groupby("Employee")["Sales"].sum())
print(df1.groupby("Employee")["Sales"].mean())
print(df1.groupby("Product")["Sales"].sum())
print(df1.groupby("Product")["Sales"].max())
print(df1.groupby("Product")["Sales"].min())
print(df1.groupby("Product")["Sales"].count())
print(df1.groupby("Product")["Sales"].agg(["sum", "mean", "max", "min", "count"]))
