from partial_substitutions import PartialSubstitution
from search import SubNode
import utils

import sympy


def check_correctness_partial(expr_true: sympy.Expr, substitution: PartialSubstitution) -> tuple[bool, sympy.Expr]:
    """
    Input:
        d: dimension of the true expression
        expr_true: the true expression
        expr_sub: substitution to be checked
        rem_idxs: indexes of all symbols that are included in the substitution, but should remain in the true expression
    """

    correct_sub = True

    # get the dimension of the true expression
    d = len(expr_true.free_symbols)

    # get the index of the y values (in a joined matrix of X and y)
    y_idx = d

    # get the substitution expression as sympy and all indexes that are included in the substitution
    expr_sub = substitution.expression.evaluate_symbolic()[0]
    subs_idxs = sorted([int(str(x).split('_')[-1]) for x in expr_sub.free_symbols if 'x_' in str(x)])

    # check if all "removed" symbols are actually included in the substitution
    removed_idxs = substitution.removed_vars
    remaining_idxs = set(subs_idxs).difference(removed_idxs)
    for idx in removed_idxs:
        if not idx in subs_idxs:
            # a removed index is missing in the substitution expression
            correct_sub = False

    try:
        if y_idx in removed_idxs:
            # in this case, expr_sub is an out-input-substitution
            print("pOIS")
            # at least two variables must be removed to reduce the dimension
            if not len(removed_idxs) >= 2:
                # at least one more variable besides y must be removed
                correct_sub = False

            # we substitute the true expression for all occurences of y and simplify the result
            z = str(expr_sub).replace(f'x_{y_idx}', f'({str(expr_true)})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # check if the simplified substitution does not depend on any repl_idxs anymore
            # also check if it still depends on all rem_idxs
            # if this passes, the substitution is correct
            free_symbols = z.free_symbols
            for s in free_symbols:
                if int(str(s).split('_')[-1]) in removed_idxs:
                    # expression still depends on a "removed" variable
                    correct_sub = False
            for rem_idx in remaining_idxs:
                if not f"x_{rem_idx}" in str(z):
                    # expression does not depend on a "remaining" variable
                    correct_sub = False

            # create new expression (give new indexes to the variables to ensure progressive numbering)
            all_remain_idxs = sorted([i for i in range(d) if i not in removed_idxs])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(all_remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)
        
        else:
            # expr_sub is an input substitution
            print(f"checking correctness of pIS {substitution}")
            # at least two variable must be removed to reduce the dimension
            if not len(removed_idxs) >= 2:
                # at least two variables must be removed
                correct_sub = False

            # we first solve the substitution for one of the symbols that should be replaced
            z_symb = sympy.Symbol('z')
            for repl_symb in list(expr_sub.free_symbols):
                # we must solve for a removed var
                index = int(str(repl_symb).split("_")[1])
                if index not in removed_idxs:
                    continue
                res = sympy.solve(expr_sub - z_symb, repl_symb)
                if len(res) == 1:
                    break
            if not len(res) == 1:
                # the substitution couldn't be solved for one of its variables
                # as a fallback, we check if the substitution is a subexpression of the formula
                if str(expr_sub) in str(expr_true):
                    print("substitution is subexpression")
                    expr_new = str(expr_true).replace(str(expr_sub), "z")
                    expr_new = sympy.sympify(expr_new)
                else:
                    raise AssertionError()

            # now, we replace all occurences of this symbol in the true expression by the solved substitution and then simplify
            z = str(expr_true).replace(str(repl_symb), f'({str(res[0])})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # check if the result does not depend on any of the variables that should be replaced
            # if this passes, the substitution is correct
            for i in removed_idxs:
                if f'x_{i}' in str(z):
                    # expression still depends on a "removed" variable
                    correct_sub = False
            for rem_idx in remaining_idxs:
                if not f"x_{rem_idx}" in str(z):
                    # expression does not depend on a "remaining" variable
                    correct_sub = False
    
            # create new expression
            # replace z with z_-1, all others with z_i
            z = z.subs(z_symb, sympy.Symbol('x_-1'))
            all_remain_idxs = sorted([int(str(s).split('_')[-1]) for s in z.free_symbols])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(all_remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)

        if correct_sub:
            return (True, expr_new)
        else:
            return (False, None)

    except AssertionError:
        return (False, None)    


def validate_path(true_expression_str: str, leaf_node: SubNode) -> list[tuple[bool, SubNode, sympy.Expr]]:
    """
    Recursively checks the correctness of the leaf substitution and all substitutions up to the root problem.
    All substitutions on this path are returned along with a bool indicating their correctness.
    """

    if leaf_node.parent == None:
        # this is the first substitution starting at the root problem
        true_expression = sympy.sympify(true_expression_str)
        return [(True, leaf_node, true_expression)]
    else:
        parents = validate_path(true_expression_str, leaf_node.parent)
        (_, _, parent_expression) = parents[-1]
        if parent_expression is None:
            return parents + [(False, leaf_node, None)]
        else:
            (sub_correct, resulting_expression) = check_correctness_partial(parent_expression, leaf_node.substitution)
            return parents + [(sub_correct, leaf_node, resulting_expression)]