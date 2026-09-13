import numpy as np

# Generate marks of 100 students
marks = np.random.randint(35, 100, 100)
print(marks)

# Boolean Masking

# students scoring above 90
print(marks > 90)
# students who failed(<35)
print(marks < 35)


# logical operator- scores b/w 60 & 80
print(marks[(marks >= 60) & (marks <= 80)])

a = marks[(marks > 75)]
print(a)
print(len(a))

# Assignment

# program1
marks1 = np.array([25, 30, 60, 75, 90, 35, 80, 95])
print(marks1[(marks1 > 70)])
print(marks1[marks1 < 40])

# program2
print(marks1[(marks1 >= 50) & (marks1 <= 90)])

# program3
arr = np.array([100, 200, 300, 400, 500, 600])
print(arr[[0, 2, 5]])

# program4
arr1 = np.array([5, 3, 8, 1, 9, 2])
print(np.sort(arr1))
print(np.sort(arr1)[::-1])

# program5
arr2 = np.array([1, 1, 2, 3, 3, 4, 8, 8, 5, 5, 6])
print(np.unique(arr2))

# AI challenge
ages = np.random.randint(18, 60, 1000)

# count people aged 18-25
aged18_25 = ages[(ages >= 18) & (ages <= 25)]
count1 = len(aged18_25)
print(count1)

# count people aged 26-40
aged26_40 = ages[(ages >= 26) & (ages <= 40)]
count2 = len(aged26_40)
print(count2)

# count people aged 41-60
aged41_60 = ages[(ages >= 41) & (ages <= 60)]
count3 = len(aged41_60)
print(count3)
