from copy import deepcopy
from itertools import combinations
import sympy
from numpy import average
import time
from datetime import datetime

from DAG_search.search import substitution_loop, SubNode
from DAG_search.correctness import validate_path, check_correctness_partial
from DAG_search.partial_substitutions import codec_coefficient
from DAG_search.substitutions import elimination_loop
from tests.testutils import get_equation, gen_dataset
from DAG_search.benchmarking import avg_reduction_rate, Measurement
from DAG_search.benchmarking import possible_reduction_rate


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

    test = (34, 36)
    feynman = (244, 374)
    eponymous = (33, 244)

    jobs: dict[str, tuple[tuple[int, int], int]] = {
        "test_1": (test, 1),
        "feynman_2": (feynman, 2),
        "feynman_3": (feynman, 3),
        "feynman_4": (feynman, 4),
        "feynman_5": (feynman, 5),
        "eponymous_2": (eponymous, 2),
        "eponymous_3": (eponymous, 3),
        "eponymous_4": (eponymous, 4),
        "eponymous_5": (eponymous, 5)
    }

    format_duration = lambda s: f"{int(s)//86400}d {(s:=int(s)%86400)//3600}h {(s:=s%3600)//60}m {s%60}s"

    results: dict[str, dict[Measurement, float]] = {}
    
    n_executions = 10

    total_start_time = time.time()

    print(f"START TIME: {datetime.fromtimestamp(total_start_time).strftime("%d.%m.%Y %H:%M:%S")}")

    for jobname, (interval, beam_len) in jobs.items():

        print(f"\n-------------------- NEXT JOB: {jobname} (beam length {beam_len}) --------------------")

        job_start_time = time.time()
        print(f"JOB START TIME: {datetime.fromtimestamp(job_start_time).strftime("%d.%m.%Y %H:%M:%S")}")

        formulas = []
        ranges = []
        reduction_dicts = []

        print("extracting formulas")
        for i in range(interval[0], interval[1]):
            (formula, var_ranges) = get_equation(i)
            formulas.append(formula)
            ranges.append(var_ranges)

        for i in range(n_executions):

            execution_start_time = time.time()

            print(f"Execution {i + 1}/{n_executions} (started {datetime.fromtimestamp(execution_start_time).strftime("%d.%m.%Y %H:%M:%S")})")
            
            reduction_dict = avg_reduction_rate(formulas, ranges, k=beam_len, n_datapoints=1000, verbose=1)

            reduction_dicts.append(reduction_dict)

        result_dict: dict[Measurement, float] = {}

        for key in Measurement:
            result_dict[key] = average([element[key] for element in reduction_dicts])

        results[jobname] = deepcopy(result_dict)

        print(f"---- RESULTS FOR JOB: {jobname} ----")
        print(f"Beam length: {beam_len}")
        for key, value in result_dict.items():
            print(f"{key}: {value}")

        job_end_time = time.time()
        job_duration = job_end_time - job_start_time

        print(f"Job duration: {format_duration(job_duration)} ({job_duration} seconds)")

    print("\n\n -------------------- ALL RESULTS --------------------")
    for job_name, job_results in results.items():
        print(f"\n ---------- JOB: {job_name} ----------")
        for key, value in job_results.items():
            print(f"{key}: {value}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nFINISH TIME: {datetime.fromtimestamp(total_end_time).strftime("%d.%m.%Y %H:%M:%S")}")
    print(f"TOTAL DURATION: {format_duration(total_duration)} ({total_duration} seconds)")

def possible_reductions_test():

    test = (255, 256)
    feynman = (244, 374)
    eponymous = (33, 244)

    intervals = [
        #test,
        feynman,
        eponymous
    ]

    for interval in intervals:

        print("\n-------------------- NEW SECTION --------------------")

        formulas: list[str] = []
        start_time = time.time()

        print("\nextracting formulas")
        for i in range(interval[0], interval[1]):
            (formula, _) = get_equation(i)
            formulas.append(formula)

        complete_full_reductions = []
        partial_full_reductions = []
        complete_red_rates = []
        partial_red_rates = []
        complete_normed_red_rates = []
        partial_normed_red_rates = []

        for index, formula in enumerate(formulas):

            print(f"\n\n[{index + 1}/{len(formulas)}] Next expression: {formula}")

            expression = sympy.sympify(formula)
            for use_partials in [False, True]:
                print(f" --- {"partial" if use_partials else "complete"} substitutions:")
                completely_reduced, red_rate, normed_red_rate = possible_reduction_rate(expression, use_partials, verbose=2)
                if use_partials:
                    partial_full_reductions.append(completely_reduced)
                    partial_red_rates.append(red_rate)
                    partial_normed_red_rates.append(normed_red_rate)
                else:
                    complete_full_reductions.append(completely_reduced)
                    complete_red_rates.append(red_rate)
                    complete_normed_red_rates.append(normed_red_rate)
                
                print(f"reduction rate: {red_rate}")
                print(f"normed reduction rate: {normed_red_rate}")
                if completely_reduced:
                    print("the formula was reduced to dimension 1")
                    if not use_partials:
                        # if the formula can be completely reduced with complete substitutions, then this is possible with partial substitutions too
                        partial_full_reductions.append(completely_reduced)
                        partial_red_rates.append(red_rate)
                        partial_normed_red_rates.append(normed_red_rate)
                        print(" --- partial substitutions are skipped")
                        break


        complete_full_reduction_percentage = average(complete_full_reductions)
        complete_avg_red_rate = average(complete_red_rates)
        complete_avg_normed_red_rate = average(complete_normed_red_rates)

        partial_full_reduction_percentage = average(partial_full_reductions)
        partial_avg_red_rate = average(partial_red_rates)
        partial_avg_normed_red_rate = average(partial_normed_red_rates)

        print("\n---- RESULTS ----")
        print(f"complete_full_reduction_percentage: {complete_full_reduction_percentage}")
        print(f"complete_avg_red_rate: {complete_avg_red_rate}")
        print(f"complete_avg_normed_red_rate: {complete_avg_normed_red_rate}")
        print(f"partial_full_reduction_percentage: {partial_full_reduction_percentage}")
        print(f"partial_avg_red_rate: {partial_avg_red_rate}")
        print(f"partial_avg_normed_red_rate: {partial_avg_normed_red_rate}")

        end_time = time.time()

        print(f"\nDuration: {end_time - start_time} seconds")