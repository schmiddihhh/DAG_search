from copy import deepcopy
from itertools import combinations
import sympy

from DAG_search.search import substitution_loop, SubNode
from DAG_search.correctness import validate_path, check_correctness_partial
from DAG_search.partial_substitutions import codec_coefficient
from DAG_search.substitutions import elimination_loop
from tests.testutils import get_equation, gen_dataset
from DAG_search.benchmarking import avg_reduction_rate


from DAG_search import dag_search, substitutions

def test_search():
    #(equation, ranges) = get_equation(249)
    (equation, ranges) = get_equation(248)
    (X, y) = gen_dataset(equation, ranges, 1000, 80204)

    subtree = substitution_loop(X, y, codec_coefficient, 5, verbose=2)

    # find the lowest loss in the whole tree
    best_substitution = SubNode(1.0, None, None)
    for beam in subtree.beams():
        for element in beam.element_list():
            element: SubNode
            if element.loss < best_substitution.loss:
                best_substitution = element

    assert best_substitution is subtree.best_substitution(), "ALARM"

    correctness_list = validate_path(equation, best_substitution)

    for (correct, subs, expr) in correctness_list:
            print(f"\t{"correct" if correct else "incorrect"}: {subs} -> {expr}")  

def runtime_substitutions():
    (equation, ranges) = get_equation(249)
    (X, y) = gen_dataset(equation, ranges, 10000, 80204)

    best_substitutions, best_Xs, best_ys = elimination_loop(X, y, codec_coefficient, verbose = 2)

    print(best_substitutions)

def benchmark():

    # generate a data set
    expression_index = 248
    (equation, ranges) = get_equation(expression_index)
    (X, y) = gen_dataset(equation, ranges, 100, 12345678)

    udfs = dag_search.DAGRegressor(k = 0, n_calc_nodes = 1)
    # udfs.fit(X, y)
    # print(udfs.model())

    subs = substitutions.SubstitutionRegressor(udfs)
    subs.fit(X, y)
    print(subs.model())

    # # for each equation: get some randomly sampled data points and try to do a regression on it
    # nonfitting = max_nonfitting = 0
    # min_nonfitting = len(equations)
    # for index, equation in enumerate(equations):

    #     # convert the equation into a callable
    #     expression: sympy.Expr = sympy.sympify(equation)
    #     free_variables: set = expression.free_symbols
    #     f = sympy.lambdify(free_variables, expression)
    #     print(f"index {index}")
    #     print(f"\nEquation: {equation} with variables {free_variables}")

    #     # generate a uniform distribution for each variable
    #     partial_ranges = ranges[index]
    #     print(equations[index])
    #     print(partial_ranges)
    #     print(ranges_str[index])
    #     if not len(free_variables) == len(partial_ranges):
    #         nonfitting += 1
    #         min_nonfitting = min(min_nonfitting, index)
    #         max_nonfitting = max(max_nonfitting, index)
    # print(nonfitting, min_nonfitting, max_nonfitting)

def dependency_test():

    # get a data set for a specific equation
    (equation, ranges) = get_equation(248)
    (X, y) = gen_dataset(equation, ranges, 100000, 123456)

    # calculate the functional dependency score on the original data
    dependency_before = 1 - codec_coefficient(X, y)
    print(f"Functional dependency score with original data: {dependency_before}")

    # apply the first (valid) partial substitution to the data (t = x3 * x5)
    X_subs_valid = deepcopy(X)
    X_subs_valid[:, 5] = X_subs_valid[:, 3] * X_subs_valid[:, 5]

    # calculate the functional dependency score on the modified data (valid substitution)
    dependency_valid = 1 - codec_coefficient(X_subs_valid, y)
    print(f"Functional dependency score with valid substitution x3 * x5: {dependency_valid}")

    # apply the second (invalid) partial substitution to the data (t = x0 * x5)
    X_subs_invalid = deepcopy(X)
    X_subs_invalid[:, 5] = X_subs_invalid[:, 0] * X_subs_invalid[:, 5]

    # calculate the functional dependency score on the modified data (invalid substitution)
    dependency_invalid = 1 - codec_coefficient(X_subs_invalid, y)
    print(f"Functional dependency score with invalid substitution x0 * x5: {dependency_invalid}")

    # iterate over all combinations
    test_values = []
    for (i0, i1) in combinations(range(6), 2):
        for i in [i0, i1]:
            X_new = deepcopy(X)
            X_new[:, i] = X_new[:, i0] * X_new[:, i1]
            dependency = 1 - codec_coefficient(X_new, y)
            string = f"Score with x{i0} * x{i1} (replacing x{i}): {dependency}"
            test_values.append((string, dependency))
    
    # rank the combinations by score and print them
    test_values.sort(key=lambda value: value[1], reverse=True)
    print("\nSubstitutions ranked by their dependency score:")
    for value in test_values:
        print(value[0])

def correctness_check_test():
    # pIS
    equation = get_equation(248)[0]
    equation = sympy.sympify(equation)
    dimension = len(equation.free_symbols)
    print(f"The equation is {equation} with dimension {dimension}")

    substitutions = [
        ("x_3*x_5", [3], "false, since only one variable is removed"),
        ("x_2-x_3*x_4", [3], "true"),
        ("x_1*x_5", [1], "false, since only one variable is removed")
    ]

    for (substitution, remaining, expected_result) in substitutions:
        print(f"\nSubstitution: {substitution} with remaining variable(s) {remaining} and dimension {dimension}\nExpected result: {expected_result}")
        substitution_sym = sympy.sympify(substitution)
        result = check_correctness_partial(equation, substitution_sym, remaining)
        print(result)

    # pOIS
    equation = "x_0*x_1*x_2+(x_0*x_1*(x_1+log(x_1)))/x_2"
    equation = sympy.sympify(equation)
    dimension = len(equation.free_symbols)
    print(f"\nThe equation is {equation} with dimension {dimension}")

    substitutions = [
        ("x_3/x_0", [], "true, since one variable is removed"),
        ("x_3/(x_0*x_1)", [1], "true, since one variable is removed"),
        ("x_3/(x_0*x_1)", [], "false, since one variable too much is removed")
    ]

    for (substitution, remaining, expected_result) in substitutions:
        print(f"\nSubstitution: {substitution} with remaining variable(s) {remaining} and dimension {dimension}\nExpected result: {expected_result}")
        substitution_sym = sympy.sympify(substitution)
        result = check_correctness_partial(equation, substitution_sym, remaining)
        print(result)

def avg_reduction_test():

    formulas = []
    ranges = []

    for i in range(252, 270):
        (formula, var_ranges) = get_equation(i)
        formulas.append(formula)
        ranges.append(var_ranges)

    avg_reduction = avg_reduction_rate(formulas, ranges, n_datapoints=1000, verbose=2)

    print(f"RESULT: {avg_reduction}")