# zeros() builtin function
import numpy as np

arr_with_zeros = np.zeros((3, 3))
print(arr_with_zeros)

# ones() builtin function
arr_with_ones = np.ones((4, 4))
print(arr_with_ones)

# full() builtin function
arr_with_value = np.full((3, 3), 5)
print(arr_with_value)

# arange() builtin function
arr_with_range = np.arange(10, 100, 10)
print(arr_with_range)

# Indexing
print(arr_with_range[3])

# convert above list of elements to 3 rows 4 colums->use reshape method
new_arr = arr_with_range.reshape((3, 3))
print(new_arr)

# eye() builtin function
arr_with_idenetity_matrix = np.eye(5)
print(arr_with_idenetity_matrix)

# Assignment Programs
# Numbers from 1-25 with arange()
arr1 = np.arange(1, 26)
print(arr1)
# convert into 5X5
new_arr1 = arr1.reshape((5, 5))
print(new_arr1)
print(new_arr1.shape)
print(new_arr1.ndim)
print(new_arr1.size)
print(new_arr1[0])
print(new_arr1[4])
print(new_arr1[:, 2])  #    : -> all rows, 2 -> columns
print(new_arr1[-1, -1])

arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2[1][1])
print(arr2[1][2])
print(arr2[2][1])
