from enum import Enum
import sympy
import numpy as np
from tqdm import tqdm

from utils import subsets
from comp_graph import CompGraph
from dag_search import DAG_Loss_fkt, exhaustive_search
from search import SubNode, substitution_loop
from partial_substitutions import codec_coefficient, PartialSubstitution
from tests.testutils import gen_dataset
from correctness import validate_path, check_correctness_partial


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

def normed_reduction_rate(best_substitution: SubNode) -> float:
    """
    Calculates the normed reduction rate of the given substitution.

    The normed reduction rate is calculated by the following formula: 1 - (#(dimension of the substitution's subproblem) / #(dimension of the original problem)).
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

    try:
        normed_reduction = 1.0 - ((dimension_subproblem - 1) / (dimension_original - 1))
    except ZeroDivisionError:
        normed_reduction = 1.0
    
    return normed_reduction

class Measurement(Enum):
    COMPLETE_REDUCTION = 1
    NORMED_COMPLETE_REDUCTION = 2
    PARTIAL_REDUCTION = 3
    NORMED_PARTIAL_REDUCTION = 4
    COMPLETE_MAX_REDUCTION_PERCENTAGE = 5
    PARTIAL_MAX_REDUCTION_PERCENTAGE = 6

def avg_reduction_rate(formulas: list[str], ranges: list[list[tuple[float, float]]], k: int = 1, n_datapoints: int = 1000, verbose: int = 0) -> dict[str, float]:
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
    formula_count = len(formulas)
    if formula_count != len(ranges):
        raise ValueError("Both lists must have the same length.")
    
    reduction_rates_complete = []
    normed_reduction_rates_complete = []
    reduction_rates_partial = []
    normed_reduction_rates_partial = []
    complete_max_reductions = 0
    partial_max_reductions = 0

    if verbose == 1:
        formulas = tqdm(formulas)

    for formula_index, (formula_str, var_ranges) in enumerate(zip(formulas, ranges)):

        if verbose >= 2:
            print(f"\n[{formula_index + 1}/{formula_count}] Next expression: {formula_str}")

        formula: sympy.Expr = sympy.sympify(formula_str)
        var_count = len(formula.free_symbols)

        # count of ranges must match the count of input variables
        if var_count != len(var_ranges):
            if verbose >= 2:
                print(f"Ranges do not match the variable count ({len(var_ranges)} ranges vs. {var_count} variables). Skipping this expression.")
            continue

        # generate a random dataset
        (X, y) = gen_dataset(formula, var_ranges, n_datapoints)

        for only_complete_subs in [True, False]:

            if verbose >= 3:
                print(f"\nCalculating with {"complete" if only_complete_subs else "partial"} substitutions")

            # find the best substitutions
            best_substitutions = None
            retry_counter = 0
            while best_substitutions is None and retry_counter < 10:
                retry_counter += 1
                try:
                    best_substitutions = substitution_loop(X, y, codec_coefficient, k=k, verbose=verbose-1, only_complete_subs=only_complete_subs)
                except ValueError:
                    # sometimes happens (depends on the random generated dataset)
                    (X, y) = gen_dataset(formula, var_ranges, n_datapoints)
            if best_substitutions is None:
                if verbose >= 1:
                    print(f"Too many exceptions for formula {formula} ({"complete" if only_complete_subs else "partial"}) - skipping this formula")
                continue
            best_substitution = best_substitutions.best_substitution()

            # get the last correct substitution in the path
            correct_substitutions = validate_path(str(formula), best_substitution, verbose=verbose-1)
            if verbose >= 3:
                print(f"Path of the given {"complete" if only_complete_subs else "partial"} substitution:")
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
            normed_reduction = normed_reduction_rate(last_correct_substitution)
            max_reduction = (normed_reduction == 1.0)
            if only_complete_subs:
                reduction_rates_complete.append(reduction)
                normed_reduction_rates_complete.append(normed_reduction)
                if max_reduction:
                    complete_max_reductions += 1
            else:
                reduction_rates_partial.append(reduction)
                normed_reduction_rates_partial.append(normed_reduction)
                if max_reduction:
                    partial_max_reductions += 1

            if verbose >= 2:
                print(f"Calculated reduction rate ({"complete" if only_complete_subs else "partial"}): {reduction} / {normed_reduction} (normed)")

    avg_reduction_complete = float(np.average(reduction_rates_complete))
    avg_normed_reduction_complete = float(np.average(normed_reduction_rates_complete))
    avg_reduction_partial = float(np.average(reduction_rates_partial))
    avg_normed_reduction_partial = float(np.average(normed_reduction_rates_partial))
    complete_max_reduction_percentage = float(complete_max_reductions) / float(len(formulas))
    partial_max_reduction_percentage = float(partial_max_reductions) / float(len(formulas))

    if verbose >= 3:
        print(f"\nReduction rates (complete): {reduction_rates_complete} / {normed_reduction_rates_complete} (normed)")
        print(f"Reduction rates (partial): {reduction_rates_partial} / {normed_reduction_rates_partial} (normed)")
    if verbose >= 2:
        print(f"\nAvg reduction rate (complete): {avg_reduction_complete} / {avg_normed_reduction_complete} (normed)")
        print(f"Avg reduction rate (partial): {avg_reduction_partial} / {avg_normed_reduction_partial} (normed)")
        print(f"% reduced to dimension 1: {complete_max_reduction_percentage} (complete) / {partial_max_reduction_percentage} (partial)")

    return_dict: dict[Measurement, float] = {
        Measurement.COMPLETE_REDUCTION: avg_reduction_complete,
        Measurement.NORMED_COMPLETE_REDUCTION: avg_normed_reduction_complete,
        Measurement.PARTIAL_REDUCTION: avg_reduction_partial,
        Measurement.NORMED_PARTIAL_REDUCTION: avg_normed_reduction_partial,
        Measurement.COMPLETE_MAX_REDUCTION_PERCENTAGE: complete_max_reduction_percentage,
        Measurement.PARTIAL_MAX_REDUCTION_PERCENTAGE: partial_max_reduction_percentage
    }

    return return_dict

def possible_reduction_rate(formula: sympy.Expr, use_partial_subs: bool, substitution_nodes: int = 1, verbose: int = 0) -> tuple[bool, float, float]:
    """
    Calculates how far the given formula can be reduced within the search space used in the substitution search.

    Parameters
    ---------
    formula : str
        The formula that will be tested.
    use_partial_subs : bool
        Controls whether partial substitutions are used for the calculations.
    verbose : int = 0
        Controls the amount of console output.
    substitution_nodes : int = 1
        Controls the maximum size of the substitutions and thus the size of the search space.

    Returns
    -------
    A tuple that consists of 3 values:
    - bool: indicates if the formula could be reduced to dimension 1
    - float: the reduction rate
    - float: the normed reduction rate

    This uses the exhaustive_search function to search over all substitutions within the search space.
    The RedRateLossFkt is used to "score" the substitutions, but in this case, it just determines if the given substitution is correct.
    For each found correct substitution, a new search is started on the subproblem, thus implementing a greedy search for substitutions that reduce the original problem iteratively.
    If a path of substitutions is found that reduces the formula to dimension 1, the search is aborted by throwing an exception. Else, the best found reduction rate is passed back to the search loop on the level above.
    """

    print(f"Calculating possible reduction rate for expression {formula}")

    original_dimension = len(formula.free_symbols)
    if original_dimension <= 1:
        # the original formula already has dimension 1
        return (True, 0.0, 1.0)
    
    # first try with boosted search
    red_rate_tester = RedRateTester(use_partial_subs, formula, substitution_nodes, verbose, True)
    red_rate_tester_2 = RedRateTester(use_partial_subs, formula, substitution_nodes, verbose, False)
    Xy = np.array([[0 for _ in range(original_dimension + 1)] for _ in range(2)])
    params = {
        'X' : Xy,
        'n_outps' : 1,
        'loss_fkt' : red_rate_tester,
        'k' : 0,
        'n_calc_nodes' : substitution_nodes,
        'n_processes' : 1,
        'topk' : 1,
        'verbose' : 2,
        'max_orders' : 10000, 
        'stop_thresh' : 1e-30
    }
    params2 = {
        'X' : Xy,
        'n_outps' : 1,
        'loss_fkt' : red_rate_tester_2,
        'k' : 0,
        'n_calc_nodes' : substitution_nodes,
        'n_processes' : 1,
        'topk' : 1,
        'verbose' : 2,
        'max_orders' : 10000, 
        'stop_thresh' : 1e-30
    }

    try:
        print("boosted search")
        exhaustive_search(**params)
        print("non-boosted search")
        exhaustive_search(**params2)
        reduced_to_1 = False
    except ReducedTo1Exception:
        reduced_to_1 = True
    
    if reduced_to_1:
        red_rate = (original_dimension - 1) / original_dimension
        return (True, red_rate, 1.0)
    else:
        best_found_dimension = red_rate_tester.best_found_dimension
        removed_dimensions = (original_dimension - best_found_dimension)
        red_rate = removed_dimensions / original_dimension
        normed_red_rate = removed_dimensions / (original_dimension - 1)
        return (False, red_rate, normed_red_rate)
    



class ReducedTo1Exception(Exception):
    pass

class RedRateTester(DAG_Loss_fkt):
    """
    Loss function placeholder that, instead of calculating a loss, calls a new search loop on all correct substitutions.
    """

    def __init__(self, use_partial_subs: bool, root_function: sympy.Expr, substitution_nodes: int, verbose: int, boosted: bool):
        '''
        Loss function that scores partial substitutions on a dataset.

        @Params:
            score_func... function that scores the dataset after applying the substitution
        '''
        super().__init__()
        self.use_partial_subs = use_partial_subs
        self.root_function = root_function
        self.root_dimension = len(self.root_function.free_symbols)
        self.substitution_nodes = substitution_nodes
        self.verbose = verbose
        self.best_found_dimension = np.inf
        self.boosted = boosted
        
    def __call__(self, _: np.ndarray, substitution:CompGraph, c: np.ndarray):
        '''
        Applies the substitution to the dataset Xy and returns the scores of all partial variations.

        For every subset of the set of variables appearing in the substitution (including empty set and 
        equal set), we calculate a separate score by removing exactly these variables from the 
        dataset. All scores are returned in a dict where they are sorted into lists, depending on 
        the dimension of the resulting problem.

        @Params:
            Xy... dataset: stacked matrix of X and y values
            substitution... DAG that represents the substitution

        @Returns:
            Loss for different constants (but ignores the constants?)
        '''

        sub_vars = substitution.evaluate_symbolic()[0].free_symbols
        sub_var_idxs = [int(str(s).split('_')[-1]) for s in sub_vars]
        is_out_input = f"x_{self.root_dimension}" in sub_vars
        partial_substitution = PartialSubstitution(substitution, is_out_input)

        if self.use_partial_subs:
            removed_var_sets = list(subsets(sub_var_idxs, minsize=2))
        else:
            removed_var_sets = [sub_var_idxs]

        for removed_vars in removed_var_sets:
            # in out-input-substitutions, the original y must always be removed
            if is_out_input and not self.root_dimension in removed_vars:
                removed_vars.append(self.root_dimension)
        
            partial_substitution.set_removed_vars(removed_vars)

            correct, res_expr = check_correctness_partial(self.root_function, partial_substitution)

            if correct:

                # get the dimension of the subproblem
                new_dimension = len(res_expr.free_symbols)

                # update the best dimension if necessary
                if new_dimension == 1:
                    # the substitution was reduced to dimension 1
                    raise ReducedTo1Exception()
                else:
                    self.best_found_dimension = min(self.best_found_dimension, new_dimension)
                
                # start a new search on the just found subproblem
                if not self.boosted or len(str(res_expr)) < len(str(self.root_function)):
                    print(f"New search iteration (after substitution {partial_substitution}, remaining formula: {res_expr} with dimension {new_dimension})")
                    new_red_rate_tester = RedRateTester(self.use_partial_subs, res_expr, self.substitution_nodes, self.verbose, self.boosted)
                    Xy = np.array([[0 for _ in range(new_dimension + 1)] for _ in range(2)])
                    params = {
                        'X' : Xy,
                        'n_outps' : 1,
                        'loss_fkt' : new_red_rate_tester,
                        'k' : 0,
                        'n_calc_nodes' : self.substitution_nodes,
                        'n_processes' : 1,
                        'topk' : 1,
                        'verbose' : 0,
                        'max_orders' : 10000, 
                        'stop_thresh' : 1e-30
                    }
                    exhaustive_search(**params)

                    self.best_found_dimension = min(self.best_found_dimension, new_red_rate_tester.best_found_dimension)

        return 1
