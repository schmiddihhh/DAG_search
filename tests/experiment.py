from DAG_search.partial_substitutions import codec_coefficient
from DAG_search import dag_search
from random import uniform
import numpy as np
import matplotlib.pyplot as plt

n_sampling_points = 1000
sampling_interval = (-10, 10)

def source_function(x: float) -> float:
    return x ** 3

def noise() -> float:
    return uniform(0, 0)

def substitution(x: float) -> float:
    return x ** 2

X = np.ndarray((n_sampling_points, 1))
y = np.ndarray((n_sampling_points))

for i in range(n_sampling_points):
    x_dp = uniform(sampling_interval[0], sampling_interval[1])
    y_dp = source_function(x_dp) + noise()
    X[i] = [x_dp]
    y[i] = y_dp

plt.scatter(X, y)
plt.show()

print("Codec coefficient vor Transformation:", 1 - codec_coefficient(X, y))
X_augmented = np.array([[xi, substitution(xi)] for xi in X])
print("Codec coefficient nach Transformation:", 1 - codec_coefficient(X_augmented, y))

# udfs = dag_search.DAGRegressor(k = 0, n_calc_nodes = 1)
# udfs.fit(X, y)
# print(udfs.model())