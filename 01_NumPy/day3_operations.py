import numpy as np

marks = np.random.randint(50, 100, 100)
print(marks)
print(np.max(marks))
print(np.min(marks))
print(np.mean(marks))
print(np.std(marks))


arr1 = np.array([10, 20, 30])
arr2 = np.array([2, 4, 6])

print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 / arr2)

salary = np.array([25000, 30000, 40000])
salary += 5000
print(salary)

random_numbers = np.random.randint(35, 100, 100)
print(np.max(random_numbers))
print(np.min(random_numbers))
print(np.mean(random_numbers))
print(np.std(random_numbers))
print(np.median(random_numbers))


matrix = np.random.randint(1, 101, (5, 5))
print(matrix)
