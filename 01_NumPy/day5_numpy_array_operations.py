import numpy as np

# program1-Matrix Multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.matmul(A, B))

# program2-Transpose
transpose_arr = np.array([[1, 2, 3], [4, 5, 6]])
print(transpose_arr.T)

# program3-Dot Product
a = np.array([2, 4, 6])
b = np.array([1, 3, 5])
print(np.dot(a, b))

# program4-hstack,vstack
c = np.array([10, 20, 30])
d = np.array([40, 50, 60])
print(np.hstack((c, d)))
print(np.vstack((c, d)))

# program5-split the arr
s = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(np.split(s, 4))
