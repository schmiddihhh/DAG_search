import sympy
from DAG_search import utils


def check_correctness_partial(d, expr_true, expr_sub, rem_idxs: list[int]):
    """
    Input:
        d: dimension of the true expression
        expr_true: the true expression
        expr_sub: substitution to be checked
        rem_idxs: indexes of all symbols that are included in the substitution, but should remain in the true expression
    """

    try:
        # get the index of the y values (in a joined matrix of X and y)
        # y should never be a remaining variable of a partial substitution
        y_idx = d
        assert y_idx not in rem_idxs, "y is a remaining variable"

        # get all indexes that are included in the substitution
        subs_idxs = sorted([int(str(x).split('_')[-1]) for x in expr_sub.free_symbols if 'x_' in str(x)])

        # check if all "remaining" symbols are actually included in the substitution and then remove them from the list
        repl_idxs = subs_idxs.copy()
        try:
            for rem_idx in rem_idxs:
                listidx = repl_idxs.index(rem_idx)
                repl_idxs.pop(listidx)
        except ValueError:
            raise AssertionError("a remaining index was missing in the substitution expression")
        print(f"replacing {repl_idxs}")

        if y_idx in repl_idxs:
            # in this case, expr_sub is an out-input-substitution
            print("pOIS")
            # at least one variable must be removed to reduce the dimension
            assert len(repl_idxs) >= 1, "not removing enough variables"

            # we substitute the true expression for all occurences of y and simplify the result
            z = str(expr_sub).replace(f'x_{y_idx}', f'({str(expr_true)})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # check if the simplified substitution does not depend on any repl_idxs anymore
            # also check if it still depends on all rem_idxs
            # if this passes, the substitution is correct
            free_symbols = z.free_symbols
            for s in free_symbols:
                assert int(str(s).split('_')[-1]) not in repl_idxs, "expression still depends on a \"removed\" variable"
            for rem_idx in rem_idxs:
                assert f"x_{rem_idx}" in str(z), "expression does not depend on a \"remaining\" variable"

            # create new expression (give new indexes to the variables to ensure progressive numbering)
            all_remain_idxs = sorted([i for i in range(d) if i not in repl_idxs])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(all_remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)
        
        else:
            # expr_sub is an input substitution
            print("pIS")
            # at least two variable must be removed to reduce the dimension
            assert len(repl_idxs) >= 2, "not removing enough variables"

            # we first solve the substitution for one of the symbols that should be replaced
            z_symb = sympy.Symbol('z')
            repl_symb = sympy.Symbol(f'x_{repl_idxs[0]}')
            res = sympy.solve(expr_sub - z_symb, repl_symb)
            assert len(res) == 1

            # now, we replace all occurences of this symbol in the true expression by the solved substitution and then simplify
            z = str(expr_true).replace(str(repl_symb), f'({str(res[0])})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # check if the result does not depend on any of the variables that should be replaced
            # if this passes, the substitution is correct
            for i in repl_idxs:
                assert f'x_{i}' not in str(z), "expression still depends on a \"removed\" variable"
            for rem_idx in rem_idxs:
                assert f"x_{rem_idx}" in str(z), "expression does not depend on a \"remaining\" variable"
            
            
            # create new expression
            # replace z with z_-1, all others with z_i
            z = z.subs(z_symb, sympy.Symbol('x_-1'))
            all_remain_idxs = sorted([int(str(s).split('_')[-1]) for s in z.free_symbols])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(all_remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)
            
    except AssertionError as e:
        print(e)
        return (False, None)
    
    return (True, expr_new)