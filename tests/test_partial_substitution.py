from DAG_search.partial_substitutions import Partial_Substitution
from DAG_search.comp_graph import sympy2dag
from sympy import sympify
from DAG_search.tests.testutils import gen_dataset
import numpy as np
import random as rnd

source = sympy2dag(sympify("(x_0+x_1+x_2)*x_3/x_4"), 3)[0]
substitution_dag = sympy2dag(sympify("x_0+x_1+x_3"), 3)[0]
substitution = Partial_Substitution(substitution_dag, [0, 2], False)
X, y = gen_dataset(source.evaluate_symbolic()[0], [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)], 1000, 12345678)
Xy = np.column_stack((X, y))

print("Xy:\n", Xy)

Xy_new = substitution.apply(Xy)

print("Xy_new:\n", Xy_new)