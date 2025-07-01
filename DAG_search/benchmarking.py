import sympy
from numpy import average

from search import SubNode, substitution_loop
from partial_substitutions import codec_coefficient
from tests.testutils import gen_dataset
from correctness import validate_path


def reduction_rate(best_substitution: SubNode) -> float:
    """
    Calculates the reduction rate of the given substitution.

    The reduction rate is calculated by the following formula: 1 - (#(dimension of the substitution's subproblem) / #(dimension of the original problem)).
    """

    if best_substitution is None:
        raise ValueError("The given best_substitution cannot be None")

    # get the node of the original problem
    original_problem = best_substitution
    while original_problem.parent is not None:
        original_problem = original_problem.parent

    # get both dimensions
    dimension_subproblem = best_substitution.dimension
    dimension_original = original_problem.dimension

    return 1.0 - (dimension_subproblem / dimension_original)

def avg_reduction_rate(formulas: list[str], ranges: list[list[tuple[float, float]]], only_complete_subs: bool, k: int = 1, n_datapoints: int = 1000, verbose: int = 0):
    """
    Calculates the reduction rate for a list of formulas and returns the average reduction rate.

    Parameters
    ----------
    formulas : list[sympy.Expr]
        A list of all formulas that should be included in the calculation.
    ranges : list[list[tuple[float, float]]]
        Includes the valid variable range for each variable of each formula.
        The list must have the same length as the "formulas" list.
        Each list that is an element of this list must have exactly as many tuples as the corresponding formula has input variables.
    verbose : int
        Controls the amount of prints this function does.
        <= 0 -> no prints
        == 1 -> progress prints
        >= 2 -> debug prints
    """

    # lists of formulas and ranges must have the same length
    if len(formulas) != len(ranges):
        raise ValueError("Both lists must have the same length.")
    
    reduction_rates = []

    for formula_str, var_ranges in zip(formulas, ranges):

        if verbose >= 1:
            print(f"\nNext expression: {formula_str}")

        formula: sympy.Expr = sympy.sympify(formula_str)
        var_count = len(formula.free_symbols)

        # count of ranges must match the count of input variables
        if var_count != len(var_ranges):
            if verbose >= 1:
                print(f"Ranges do not match the variable count ({len(var_ranges)} ranges vs. {var_count} variables). Skipping this expression.")
            continue

        # generate a random dataset
        (X, y) = gen_dataset(formula, var_ranges, n_datapoints)

        # find the best substitutions
        best_substitutions = substitution_loop(X, y, codec_coefficient, k=k, verbose=verbose, only_complete_subs=only_complete_subs)
        best_substitution = best_substitutions.best_substitution()

        # get the last correct substitution in the path
        correct_substitutions = validate_path(str(formula), best_substitution)
        if verbose >= 2:
            print("Path of the given substitution:")
            for correct, substitution, result in correct_substitutions:
                print(f"\t{substitution.substitution} -> {correct} (result: {result})")

        (first_correct, last_correct_substitution, _) = correct_substitutions[0]
        assert first_correct
        for current_correct, current_substitution, _ in correct_substitutions:
            if not current_correct:
                break
            last_correct_substitution = current_substitution
        
        # get the reduction rate and append it to the list
        reduction = reduction_rate(last_correct_substitution)
        reduction_rates.append(reduction)

        if verbose >= 1:
            print(f"Calculated reduction rate: {reduction}")

    avg_reduction = float(average(reduction_rates))

    if verbose >= 2:
        print(f"Reduction rates: {reduction_rates}")
    if verbose >= 1:
        print(f"Avg reduction rate: {avg_reduction}")

    return avg_reduction
        