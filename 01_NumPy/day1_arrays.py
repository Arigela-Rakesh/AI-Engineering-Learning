import numpy as np

# creating 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)

# creating 2D array
arr2d = np.array([[3, 2, 1, 4, 5], [6, 7, 8, 9, 1]])
print(arr2d)

# creating 3D array

arr3d = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
print(arr3d)

print(arr3d.ndim)  # number of dimensions
print(arr3d.shape)  # rows & columns of array
print(arr3d.size)  # total no.of elements of array
print(arr3d.dtype)  # datatype of array
