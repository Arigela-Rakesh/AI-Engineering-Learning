from sklearn.metrics import mean_squared_error

actual = [10, 20, 30]

predicted = [12, 18, 33]

mse = mean_squared_error(actual, predicted)

print(mse)
