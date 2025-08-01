import pandas as pd
import numpy as np
import sympy
import random as rnd


def get_equation(index: int) -> tuple[str, list[tuple[float, float]]]:

    #print(f"Extracting the {index} th equation with its variable ranges from the dataset")

    # extract the equation from the dataset
    df = pd.read_csv("~/Uni/ba/datasets/train.csv")
    df_numpy = df.to_numpy()
    equation = df_numpy[index, 4]

    # extract the valid variable ranges for the equation
    range_str: str = df_numpy[index, 7]
    range_str = range_str[2:-2]
    partial_ranges_str = range_str.split("], [")
    partial_ranges = []
    for partial_range_str in partial_ranges_str:
        range_values = partial_range_str.split(", ", 1)
        partial_ranges.append([float(range_values[0]), float(range_values[1])])

    return (equation, partial_ranges)

def gen_dataset(equation: str, var_ranges: list[tuple[int, int]], sample_size: int, seed: int = None, verbose: int = 0) -> tuple[np.ndarray, np.ndarray]:

    # convert the equation into a sympy expression
    expression: sympy.Expr = sympy.sympify(equation)
    free_variables: set = sorted(expression.free_symbols, key=lambda s: s.name)
    f = sympy.lambdify(free_variables, expression)

    if verbose >= 1:
        print(f"Generating {sample_size} data points for equation {equation} with variables {free_variables}")

    # check if the free variable count matches the range count
    assert(len(free_variables) == len(var_ranges))

    # initialize rnd with a seed
    if seed is not None:
        rnd.seed(seed)

    # get some random data points
    noise = 0
    X = []
    y = []
    for _ in range(sample_size):
        x = []
        for var_range in var_ranges:
            x.append(rnd.uniform(var_range[0], var_range[1]))
        X.append(x)
        y_val = f(*x) + noise
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        print("ALAAAAARM: generated dataset contains non-finite values")

    return (X, y)

