from __future__ import annotations

from DAG_search.dag_search import DAG_Loss_fkt, exhaustive_search
from DAG_search.comp_graph import CompGraph
from utils import subsets

from partial_substitutions import PartialSubstitution, codec_coefficient

import warnings
import numpy as np
from collections import defaultdict
from copy import deepcopy
import heapq


class Beam:
    """
    Represents a beam in beam search.

    Attributes
    ----------
    maxlen : int
        The highest allowed count of elements in the beam.
    length : int
        The current count of elements in the beam.
    elements : list
        A list of the elements, sorted by their losses.
        This list is controlled by a heap datastructure (via heapq).
    """

    def __init__(self, maxlen: int):
        """
        Initializes the object's attributes.

        Parameters
        ----------
        maxlen : int
        """
        self.maxlen: int = maxlen
        self.length: int = 0
        self.elements: list[tuple[float, object]] = []

    def insert(self, element, loss: float):
        """
        Inserts the element into the list of elements.

        Parameters
        ----------
        element : Any
            The element to insert.
        loss : float
            The loss that was calculated for this element (outside of the Beam class).
            If the loss is higher than the losses of any element in the elements list, the current element will be dropped and not inserted.
        """
        # we invert the loss since heapq always removes the smallest item first (we need to remove the highest loss first)
        loss_inverted = - loss

        if not any(loss == loss_inverted for loss, _ in self.elements):
            # this loss value is not yet present in the heap
            if self.length < self.maxlen:
                # the element is inserted regardless of its loss, since the maximum beam size was not reached yet
                heapq.heappush(self.elements, (loss_inverted, element))
                self.length += 1
            else:
                # the element is inserted along with removing the worst scoring element
                heapq.heappushpop(self.elements, (loss_inverted, element))

    def element_list(self) -> list[object]:
        """
        Returns an unordered list of all current elements.
        """
        return [element for _, element in self.elements]

    def loss_element_list(self) -> list[tuple[float, object]]:
        """
        Returns loss-element tuples in a list, sorted by the losses.
        """
        return sorted([(- loss_inverted, element) for loss_inverted, element in self.elements], key=lambda x: x[0])
    
    def __repr__(self) -> str:
        repr_str = f"\nBeam with length {self.length}: \n"
        for element in self.element_list():
            repr_str += repr(element) + "\n"
        return repr_str[:-1]    # removes newline at the end
    
    def __str__(self) -> str:
        return self.__repr__()
    
    # def __iter__(self):
    #     for _, element in self.elements:
    #         yield element


class SubNode:
    """
    A node in a SubBeamTree.

    Attributes
    ----------
    loss : float
        The loss that was calculated for the substitution in the node.
    substitution : PartialSubstitution
        Partial substitution that is wrapped in this node.
    dataset_after : np.ndarray
        Dataset that resulted from applying the substitution to the parent dataset.
    parent : SubNode
        The previous substitution or the root problem.
    """

    def __init__(self, loss: float, substitution: PartialSubstitution, dataset_after: np.ndarray, parent: SubNode = None):
        """
        Initializes the object's attributes.

        Parameters
        ----------
        loss : float
        substitution : PartialSubstitution
        dataset_after : np.ndarray
        parent : SubNode
        """
        self.loss: float = loss
        self.substitution: PartialSubstitution = substitution
        self.dataset_after: np.ndarray = dataset_after
        self.parent: SubNode = parent
        self.dimension: int = self.dataset_after.shape[1] - 1

    def __str__(self):
        try:
            current_node = self
            repr_str = str()
            while True:
                if current_node.substitution is None:
                    repr_str += "[root]"
                else:
                    repr_str += f"{current_node.loss}: "
                    repr_str += str(current_node.substitution.expression.evaluate_symbolic()[0])
                    repr_str += f" {current_node.substitution.removed_vars}"
                current_node = current_node.parent
                if current_node is None:
                    break
                repr_str += " <- "
            return repr_str
        except Exception as e:
            return str(f"[non-evaluable SubNode: {e}]")
        
    def __repr__(self):
        return self.__str__()
    

class SubDict:
    """
    Holds a set of Beams, one for each target dimension.

    The only functionality is inserting substitutions into the data structure. A SubDict is easily mergeable into a SubBeamTree.

    Attributes
    ----------
    root : SubNode
        The problem that all substitutions in this data structure will be derived from.
    beam_len : int
        Length of the beams.
    dict : defaultdict[int, Beam]
        Dictionary that holds one Beam per dimension.
    """

    def __init__(self, beam_len: int, root: SubNode):
        """
        Init the object's attributes.
        """
        self.root: SubNode = root
        self.beam_len: int = beam_len
        self.dict: dict[int, Beam] = defaultdict(lambda: Beam(self.beam_len))

    def insert(self, dimension: int, substitution: PartialSubstitution, loss: float):
        """
        Inserts the given substitution into the according Beam.

        Parameters
        ----------
        dimension : int
            Dimension of the substitution's subproblem.
        substitution : PartialSubstitution
            Substitution to insert.
        loss : float
            Calculated loss of the substitution.
        """
        self.dict[dimension].insert(substitution, loss)

    def min_loss(self, dimension: int) -> float:
        """
        Returns the best (lowest) loss that was achieved in the given dimension.

        Parameters
        ----------
        dimension : int
            Target dimension of the substitutions that are included.
        """
        # print([loss for loss, _ in self.dict[dimension].loss_element_list()])
        return min([loss for loss, _ in self.dict[dimension].loss_element_list()]) 


class SubBeamTree:
    """
    Represents a search tree with fixed beam length.
    """

    def __init__(self, beam_len: int, root_problem: SubNode):
        self.beam_len: int = beam_len
        self.dict: dict[int, Beam] = defaultdict(lambda: Beam(self.beam_len))
        root_dimension = root_problem.dimension
        self.dict[root_dimension].insert(root_problem, root_problem.loss)

    def insert(self, node: SubNode):
        self.dict[node.dimension].insert(node, node.loss)

    def get_elements(self, dimension: int) -> list[SubNode]:
        return self.dict[dimension].element_list()

    def merge(self, sub_dict: SubDict):
        """
        Merges a SubDict into this SubBeamTree by creating SubNodes for each substitution in the SubDict, linking them to the root node of the SubDict and inserting them into self.
        """
        root_node = sub_dict.root
        for dimension, beam in sub_dict.dict.items():
            for loss, substitution in beam.loss_element_list():
                substitution: PartialSubstitution
                dataset_after = substitution.apply(root_node.dataset_after)
                new_node = SubNode(loss, substitution, dataset_after, root_node)
                self.insert(new_node)

    def beams(self) -> list[Beam]:
        return list(self.dict.values())
    
    def best_substitution(self) -> SubNode:
        """
        Returns the best scoring substitution in the whole tree.
        """
        best_loss = 1
        best_substitution = None
        for beam in self.dict.values():
            for substitution in beam.element_list():
                substitution: SubNode
                if substitution.loss < best_loss:
                    best_loss = substitution.loss
                    best_substitution = substitution
        
        return best_substitution

    

class Memory_Loss_Fkt(DAG_Loss_fkt):
    """
    Loss function that memorizes all substitutions it scored.
    """

    def __init__(self, score_func, only_complete_subs: bool, root_node: SubNode, beam_len : int = 5):
        '''
        Loss function that scores partial substitutions on a dataset.

        @Params:
            score_func... function that scores the dataset after applying the substitution
        '''
        super().__init__()
        self.score_func = score_func
        self.best_substitutions: SubDict = SubDict(beam_len, root_node)
        self.only_complete_subs = only_complete_subs
        self.total_scored_substitutions = 0
        
        
    def __call__(self, Xy:np.ndarray, substitution:CompGraph, c: np.ndarray):
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # this will store the minimum loss among all scored substitutions
            min_loss = np.inf

            # this is needed to get the function working in the elimination loop
            if len(c.shape) == 2:
                vec = True
            else:
                vec = False

            # get some info about the substitution
            expr = substitution.evaluate_symbolic()[0]
            idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])
            var_count = len(idxs)   # number of vars in the substitution
            current_dimension = Xy.shape[1] - 1
            out_input = (current_dimension in idxs) # determines if this is an out-input substitution

            # init the substitution object
            partial_substitution = PartialSubstitution(substitution, out_input)
            partial_substitution.calc_fx(Xy)

            if np.all(np.isfinite(partial_substitution.fx)):
                
                if 1 < var_count < Xy.shape[1]:   # the DAG has more than one input variable and not more than the data matrix
                    if self.only_complete_subs:
                        Xy_new = partial_substitution.apply(Xy)
                        # calculate the loss of the substitution
                        loss = self.score_func(Xy_new[:, :-1], Xy_new[:, -1])
                        # insert the substitution into the BeamDict
                        dimension_new = Xy_new.shape[1] - 1
                        self.best_substitutions.insert(dimension_new, deepcopy(partial_substitution), loss)
                        self.total_scored_substitutions += 1
                        # get the minimal loss of all tested substitutions (as the return value of this function)
                        min_loss = min(min_loss, loss)
                    else:
                        #print(f"partial and complete subs ({len(idxs)} variables)")
                        for removed_vars in subsets(idxs, minsize=2):
                            # TODO: ALAAAAARM - was mit Out-Input-Substitutionen???
                            #print(f"subset (size {len(removed_vars)})")
                            if not len(removed_vars) == len(idxs):  # else, the substitution would be a complete substitution
                                # calculate the loss of the substitution
                                partial_substitution.set_removed_vars(removed_vars)
                                Xy_new = partial_substitution.apply(Xy)
                                loss = self.score_func(Xy_new[:, :-1], Xy_new[:, -1])
                                # insert the substitution into the BeamDict
                                dimension_new = Xy_new.shape[1] - 1
                                self.best_substitutions.insert(dimension_new, deepcopy(partial_substitution), loss)
                                self.total_scored_substitutions += 1
                                # get the minimal loss of all tested substitutions (as the return value of this function)
                                min_loss = min(min_loss, loss)
        if not vec:
            return min_loss
        else:
            return [min_loss]
        

def top_k_substitutions(Xy: np.ndarray, score_fkt, root_node: SubNode, only_complete_subs: bool, k: int = 5, verbose: int = 0, substitution_nodes: int = 1) -> SubDict:
    """
    Searches over all DAGs up to the given size (substitution nodes) to find the best scoring substitutions for the given dataset.

    Parameters
    ----------
    X : np.ndarray
        An array of feature vectors (2D matrix). First dimension needs to match the length of y.
    y : np.ndarray
        An array of labels (1D vector). Has to have the same length as the first dimension of X.
    score_fkt : Any
        Callable that scores a dataset.
    k : int
        The k best scoring substitutions (per dimension) are returned.
    verbose : int
        Controls if there will be console output.
    substitution_nodes : int
        Determines the maximum node count of the substitution DAGs.
    """

    # define the loss function that also contains a store for the best substitutions
    loss_fkt = Memory_Loss_Fkt(score_fkt, True, root_node, k)

    # search over all complete substitutions with a certain size
    # the results are stored in the loss function
    params = {
        'X' : Xy,
        'n_outps' : 1,
        'loss_fkt' : loss_fkt,
        'k' : 0,
        'n_calc_nodes' : substitution_nodes,
        'n_processes' : 1,
        'topk' : 1,
        'verbose' : verbose,
        'max_orders' : 10000, 
        'stop_thresh' : 1e-30
    }
    exhaustive_search(**params)

    #print(f" - complete substitutions: {loss_fkt.total_scored_substitutions} substitutions checked (best substitutions: {[(dim, beam.element_list()) for (dim, beam) in loss_fkt.best_substitutions.dict.items()]})")

    prev_dimension = root_node.dimension
    min_loss = loss_fkt.best_substitutions.min_loss(prev_dimension - 1)

    if not only_complete_subs and not min_loss < root_node.loss:
        # try again with partial substitutions
        if verbose >= 2:
            print("probiere partielle")
        loss_fkt_2 = Memory_Loss_Fkt(score_fkt, False, root_node, k)
        params = {
            'X' : Xy,
            'n_outps' : 1,
            'loss_fkt' : loss_fkt_2,
            'k' : 0,
            'n_calc_nodes' : substitution_nodes,
            'n_processes' : 1,
            'topk' : 1,
            'verbose' : verbose,
            'max_orders' : 10000, 
            'stop_thresh' : 1e-30
        }
        exhaustive_search(**params)

        #print(f" - partial substitutions: {loss_fkt_2.total_scored_substitutions} substitutions checked (best substitutions: {loss_fkt_2.best_substitutions.dict.items()})")

        return loss_fkt_2.best_substitutions

    return loss_fkt.best_substitutions


def substitution_loop(X: np.ndarray, y: np.ndarray, score_fkt, k: int = 5, verbose: int = 0, substitution_nodes: int = 1, only_complete_subs: bool = False) -> SubBeamTree:
    

    # get the data for the root problem
    root_Xy = np.column_stack([X, y])
    root_loss = score_fkt(X, y)
    dimension = root_Xy.shape[1] - 1

    # set up the loop for the first iteration
    root_problem = SubNode(root_loss, None, root_Xy)
    best_substitutions: SubBeamTree = SubBeamTree(k, root_problem)

    while dimension >= 2:

        for parent_node in best_substitutions.get_elements(dimension):

            parent_Xy = parent_node.dataset_after

            # this executes the search and gives us a dictionary with the top k found substitutions per target dimension
            topk_subdict = top_k_substitutions(parent_Xy, score_fkt, parent_node, only_complete_subs, k, verbose, substitution_nodes)

            # get the k best substitutions for each target dimension that was covered in the search
            # insert these into the overall beams
            best_substitutions.merge(topk_subdict)

        # prepare for the next iteration: reduce dimension by 1
        dimension -= 1

        if verbose >= 2:
            print(f"\nNew iteration completed (new dimension: {dimension}):")
            for substitution in best_substitutions.get_elements(dimension):
                print(substitution)
            print()

    return best_substitutions