from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from DAG_search.comp_graph import CompGraph


class PartialSubstitution:
    """
    Representation of a partial substitution.

    Attributes
    ----------
    expression : CompGraph
        The expression of the substitution, represented by a CompGraph.
        The variables of the expression should be named x_i, where i is replaced by the according indexes.
    out_input : bool
        Determines if the substitution is handled as an input substitution (False) or an out-input substitution (True).
    removed_vars : list[int]
        The indices of all variables that should be removed from the dataset when applying the substitution.
        All indices in this list must appear in the expression too.
    vars : list[int]
        The indices of all variables that appear in the expression.
    """

    def __init__(self, expression: CompGraph, out_input: bool, removed_vars: list[int] = None):
        """
        Sets the object attributes.
        For performance reasons, the parameters are not checked for validity.

        Parameters
        ----------
        expression : CompGraph
        out_input : bool
        removed_vars : list[int]
        """
        self.expression = expression
        self.out_input = out_input
        # extract the variable indices from the expression
        self.vars = sorted([int(str(s).split('_')[-1]) for s in expression.evaluate_symbolic()[0].free_symbols])
        if removed_vars is not None:
            self.removed_vars = removed_vars
        else:
            self.removed_vars = self.vars

    def set_removed_vars(self, removed_vars: list[int]):
        """
        Sets self.removed_vars to the given value.

        Parameters
        ----------
        removed_vars : list[int]
            Value that is assigned to self.removed_vars.
        """
        self.removed_vars = removed_vars

    def apply(self, Xy: np.ndarray) -> np.ndarray:
        """
        Applies this substitution to the given dataset and returns the resulting dataset.

        Parameters
        ----------
        Xy : np.ndarray
            The dataset this substitution should be applied on.
        """
        # calculate the substitution value for each data point
        fx = self.expression.evaluate(Xy, c = np.array([]))
        if not np.all(np.isfinite(fx)):
            raise ValueError("The application of the substitution on the dataset lead to non-finite results.")

        # add the results to the dataset
        if self.out_input:
            # out-input substitution with a new y - replace the old y column with the new one
            Xy_new = np.column_stack([Xy[:, i] for i in range(Xy.shape[1] - 1) if i not in self.removed_vars] + [fx])
        else:
            # input substitution - insert the new variable to the first column
            Xy_new = np.column_stack([fx] + [Xy[:, i] for i in range(Xy.shape[1]) if i not in self.removed_vars])

        return Xy_new
    
    def __repr__(self):
        try:
            return str(f"{self.expression.evaluate_symbolic()[0]} [{self.removed_vars}]")
        except:
            return str("[non-evaluable pS]")
        
    def __str__(self):
        return self.__repr__()
    

# def translate_back(expr, transl_dict):
#     '''
#     Given an expression and a translation dict, reconstructs the original expression.

#     @Params:
#         expr... sympy expression
#         transl_dict... translation dictionary, 
#     '''
#     if len(transl_dict) == 0:
#         return expr

#     idxs = sorted([int(str(x).split('_')[-1]) for x in expr.free_symbols if 'x_' in str(x)])
#     x_expr = str(expr).replace('x_', 'z_')
#     for i in idxs:
#         x_expr = x_expr.replace(f'z_{i}', f'({transl_dict[i]})')
#     y_expr = transl_dict[len(transl_dict) - 1]
#     total_expr = f'g - ({y_expr})' # g is placeholder for rest of expression
#     total_expr = sympy.sympify(total_expr)

#     y_symb = sympy.Symbol('y')
#     res = sympy.solve(total_expr, y_symb)
#     assert len(res) > 0
#     return [sympy.sympify(str(r).replace('g', f'({x_expr})')) for r in res]


def codec_coefficient(X: np.ndarray, y: np.ndarray, k: int = 1, normalize: bool = True):
    """
    Calculates the CODEC coefficient that is used as a loss function for partial substitutions.

    The CODEC coefficient turns out a value between 0 and 1, indicating how strong the functional
        dependency between features and labels is. Here, it is used as a heuristic measure
        for the quality of partial substitutions in the search algorithm.

    Parameters
    ----------
    X : np.ndarray
        Matrix of features.
    y : np.ndarray
        Vector of labels.
    k : int
    normalize : bool
    """

    if normalize:
        z = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    else:
        z = X
    n = y.shape[0]
    
    r = y.argsort().argsort()
    l = n - r - 1
    denom = np.sum(l * (n-l))

    neigh = NearestNeighbors(n_neighbors= k+1 ).fit(z)
    nn = neigh.kneighbors(z, return_distance = False)
    
    num = np.sum(n * np.min(r[nn], axis = 1) - l**2)
    return 1- num/denom