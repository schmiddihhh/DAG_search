from DAG_search.partial_substitutions import Partial_Substitution, Beam
import random as rnd

def test_beam(n_elements: int, seed: int = None):
    if seed is not None:
        rnd.seed(seed)

    beam = Beam(5)
    losses = []
    
    for i in range(n_elements):
        loss = rnd.randint(0, 10000) / 10000
        losses.append(loss)
        assert(loss <= 1 and loss >= 0)
        substitution = Partial_Substitution(None, None, False)
        # insert the substitution twice
        # the second one should always be skipped because of the identical loss
        beam.insert(substitution, loss)
        beam.insert(substitution, loss)

    losses_sorted = sorted(list(set(losses)))   # remove duplicates and sort
    # print(losses_sorted)
    elements = beam.element_list()
    # for element in elements:
    #     print(element)

    # compare
    for i, (loss, _) in enumerate(elements):
        assert(losses_sorted[i] == loss)

    

test_beam(1000)