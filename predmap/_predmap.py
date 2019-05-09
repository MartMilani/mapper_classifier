import numpy as np
from predmap.disambiguated_node import DisambiguatedNode
import copy
import networkx as nx
import matplotlib.pyplot as plt


def float_eq(x, y, tol=1e-8):
    return x < y+tol and x > y-tol


class BinaryClassifier():
    """Class responsible for the implementation on top of a mapper graph of the binary
    classification task as designed by Francesco Palma and Thomas Boys

    Example of use:

        >>> from numpy import genfromtxt
        >>> import lmapper as lm
        >>> from lmapper.filter import Projection
        >>> from lmapper.cover import BalancedCover
        >>> from lmapper.cluster import Linkage
        >>> import predmap as mapp
        >>>
        >>> x = genfromtxt('synthetic_dataset/synthetic.csv', delimiter=',')
        >>> filter = Projection(ax=2)
        >>> cover = BalancedCover(nintervals=20,
        >>>                       overlap=0.4)
        >>> cluster = Linkage(method='single')
        >>> mapper = lm.Mapper(data=x,
        >>>                    filter=filter,
        >>>                    cover=cover,
        >>>                    cluster=cluster)
        >>> mapper.fit()
        >>>
        >>> predictor = mapp.BinaryClassifier(mapper=mapper,
        >>>                                   response_values=y,
        >>>                                   _lambda=0.4)
        >>> predictor.fit()
        >>> x0 = [0.3, 0.4, 0.7]
        >>> predictor.predict(x0)

    """
    def __init__(self, mapper, response_values, _lambda=0.2, a=0.5, beta=1, maxiter=1):
        """Init function

        Args:
            mapper (mp.Mapper): mapper object
            response_values (np.ndarray): array that contains the binary labels of the
                mapper.data matrix
            _lambda (float): constant that decides how many intervals to switch
            maxiter (int): number of iterations

        """
        self._mapper = mapper
        self._y = response_values
        self._maxiter = maxiter
        self.partition = {}
        self._lambda = _lambda
        self._a = a
        self._beta = beta
        self._graph = copy.deepcopy(mapper.complex._graph)

    def _disambiguate(self, verbose):
        """Performs the Disambiguation step.
        Instantiates the DisambiguatedNode objects on top of Node objects and put them in
        the self.partition dict.

        NOTE:
            It adds to the self.partition dict only the non-empty supernodes.

        """
        def _findduplicatednodes(verbose):
            """"""
            duplicatednodes = set()
            alreadydone = []
            for fiber in self._mapper.cover:
                intersectingfibersids = self._mapper.cover.intersecting_dict[fiber._fiber_index]
                for fiber2 in self._mapper.cover:
                    if fiber2._fiber_index in intersectingfibersids:
                        for node in fiber:
                            for node2 in fiber2:
                                el = sorted([node._id, node2._id])
                                if el not in alreadydone:
                                    if len(node._labels) == len(node2._labels):
                                        if np.array_equal(node._labels, node2._labels):
                                            # if np.asarray([x == y for x, y in zip(node._labels, node2._labels)]).all():
                                            duplicatednodes.add(el[1])
                                            # since el is sorted, and we already checked
                                            # if we already controlled el or not,
                                            # el[1] will alway add a different node
                                    alreadydone.append(el)

            if verbose:
                print("\nFound {} identical nodes in the mapper graph".format(len(duplicatednodes)))

            return list(duplicatednodes)

        def _disambiguateintersections(verbose):
            """Helper function.
            Provided that each supernode has been instantiated with its unique pointlabels
            it assign to the correct supernode according to the Single Linkage function
            (min of the dinstances) each point belonging to some intersection
            """
            intersectiondict = self._mapper.complex._intersection_dict
            data = self._mapper.data
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                for simplex in intersectiondict.keys():
                    if (len(simplex) == 2 and (simplex[0] == nodeid or (simplex[1] == nodeid and simplex[0] not in self.partition.keys()))):  # in this way we do it only once! good!
                        try:
                            second_supernode = self.partition[simplex[1]]  # could fail if it was empty!
                            for x_label in intersectiondict[simplex]:
                                x = data[x_label]

                                # applying mapper-assignment rule, that depends on the cluster method
                                if self._mapper.cluster._method == 'single':
                                    distance1 = np.min([np.linalg.norm(x-b) for b in data[supernode._pointlabels]])
                                    distance2 = np.min([np.linalg.norm(x-b) for b in data[second_supernode._pointlabels]])
                                elif self._mapper.cluster._method == 'average':
                                    distance1 = np.mean([np.linalg.norm(x-b) for b in data[supernode._pointlabels]])
                                    distance2 = np.mean([np.linalg.norm(x-b) for b in data[second_supernode._pointlabels]])
                                elif self._mapper.cluster._method == 'complete':
                                    distance1 = np.max([np.linalg.norm(x-b) for b in data[supernode._pointlabels]])
                                    distance2 = np.max([np.linalg.norm(x-b) for b in data[second_supernode._pointlabels]])

                                if distance1 < distance2:
                                    supernode._extrapointlabels.append(x_label)
                                else:
                                    second_supernode._extrapointlabels.append(x_label)
                        except KeyError:
                            for x_label in intersectiondict[simplex]:
                                supernode._extrapointlabels.append(x_label)

            # just for printing diagnostics
            tot_disambiguated_mapper_graph = 0

            # finalizing disambiguation
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                supernode._extrapointlabels = np.asarray(supernode._extrapointlabels, dtype=int)
                supernode._pointlabels = np.append(supernode._pointlabels, supernode._extrapointlabels)
                supernode._pointlabels = np.unique(supernode._pointlabels)
                supernode._size = len(supernode._pointlabels)
                if verbose:
                    print('\nDisambiguating node {}'.format(nodeid))
                    print('Original size: {}'.format(len(supernode._node._labels)))
                    print('Disambiguated size: {}'.format(supernode._size))
                    tot_disambiguated_mapper_graph += supernode._size
            if verbose:
                print('\nTotal number of disambiguated nodes: {}'.format(len(self.partition)))
                print('\nTotal number of points in the original Mapper graph: {}'.format(np.alen(self._mapper.data)))
                print('\nTotal number of points in the disambiguated Mapper graph: {}'.format(tot_disambiguated_mapper_graph))

        def _initiatesupernodes(verbose):
            """Helper function
            Provided that each DisambiguatedNode has been instantiated in self.partition and the
            _disambiguateintersections() has been run,
                - it finds the corresponding responsevalues
                - it sets the majorityvotes per each supernode
            """
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                if supernode._size:
                    supernode._findresponsevalues(self._y)
                    supernode._setmajorityvote()

        # first, taking care of duplicated nodes by
        # 1) finding them
        # 2) eliminating entries from self._mapper.complex._intersection_dict
        #    such that _disambiguateintersections() will find a unique node
        duplicatednodes = _findduplicatednodes(verbose)
        for nodeid in duplicatednodes:
            to_eliminate = []
            for simplex in self._mapper.complex._intersection_dict.keys():
                if nodeid in simplex:
                    to_eliminate.append(simplex)
            for simplex in to_eliminate:
                self._mapper.complex._intersection_dict.pop(simplex)

        for fiber in self._mapper.cover:
            for node in fiber:
                if node._id not in duplicatednodes:
                    supernode = DisambiguatedNode(node, fiber, self._mapper)
                    supernode._finduniquelabels(self._y)
                    # initialize unique labels, supernode._onlyones and supernode._onlyzeros

                    # --------------- WARNING ----------------
                    # THESE LINES THAT FOLLOW IMPLEMENT A SOLUTION TO THE FOLLOWING PROBLEMS:
                    #
                    # Problem 1)
                    #   a node is a subset of another node, thus all its datapoints
                    #   are shared with the second node
                    # Problem 2)
                    #   node_1 = {1,2}
                    #   node_2 = {2,3}
                    #   node_3 = {3,4}
                    #   node_4 = {4,5}
                    #   in other words, node_2 and node_3 contain only shared points
                    #   AND have non-empty intersection
                    #
                    # This naive solution works only for one-dimensional filters and with a
                    # uniform cover with percentage_overlap < 50%.
                    # TODO: find a generalization of this solution!
                    #
                    if supernode._size:
                        self.partition[node._id] = supernode
                    else:
                        to_eliminate = []
                        for simplex in self._mapper.complex._intersection_dict.keys():
                            if node._id in simplex:
                                to_eliminate.append(simplex)
                        if len(to_eliminate) >= 2:
                            for simplex in to_eliminate:
                                if node._id == simplex[0]:
                                    self._mapper.complex._intersection_dict.pop(simplex)
                    # -------------------------------------------
        _disambiguateintersections(verbose)
        _initiatesupernodes(verbose)

    def _computeintervals(self, verbose):
        """Helper function called by fit()
        """
        if verbose:
            print('\nComputing intervals to switch')
        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            if supernode._pure:
                continue
            else:
                supernode._computeintervals(self._beta, verbose)

    def _applyscorefunction(self, verbose):
        """Helper function called by fit()
        """

        if verbose:
            print('\nComputing score function')

        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            if supernode._pure:
                supernode._score = 0
            else:
                supernode._applyscorefunction(self._a)

        if verbose:
            score_list = []
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                score_list.append(supernode._score)
            print(score_list)
            print('S_min = {}, S_max = {}'.format(np.min(score_list), np.max(score_list)))

    def _intervals_to_flip(self, verbose):
        """Helper function called by fit()
        """
        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            if supernode._pure:
                continue
            else:
                supernode._intervals_to_flip(self._lambda, verbose)

    def _assign(self, x, fx, leave_one_out=False, index=None):
        """Helper function called by _predict()
        Finds the DisambiguatedNode that the point x belongs to

            Args:
                x (np.ndarray): one dimensional ndarray representing one data point
                fx (float): filter value of x
        """
        data = self._mapper.data
        old_d = float("inf")
        toreturn = None
        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            if fx >= supernode._fiber._filter_minima and fx <= supernode._fiber._filter_maxima:

                if leave_one_out:
                    _pointlabels = supernode._pointlabels[supernode._pointlabels != index]
                    if not len(_pointlabels):
                        return supernode
                else:
                    _pointlabels = supernode._pointlabels

                points = data[_pointlabels]
                # implementing mapper-assignment
                if self._mapper.cluster._method == 'single':
                    d = np.min([np.linalg.norm(x-b) for b in points])
                if self._mapper.cluster._method == 'average':
                    d = np.mean([np.linalg.norm(x-b) for b in points])
                if self._mapper.cluster._method == 'complete':
                    d = np.max([np.linalg.norm(x-b) for b in points])

                if d < old_d:
                    toreturn = supernode
                    old_d = d
        if toreturn:
            return toreturn

        old_d = float("inf")
        toreturn = None
        max_fx = -np.float('inf')
        min_fx = np.float('inf')
        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            if supernode._fiber._filter_minima <= min_fx:
                min_fx = supernode._fiber._filter_minima
            if supernode._fiber._filter_maxima >= max_fx:
                max_fx = supernode._fiber._filter_maxima
        # once I found the minima and maxima, let's assign it
        if fx <= min_fx:
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                if float_eq(supernode._fiber._filter_minima, min_fx):

                    if leave_one_out:
                        _pointlabels = supernode._pointlabels[supernode._pointlabels != index]
                        if not len(_pointlabels):
                            return supernode
                    else:
                        _pointlabels = supernode._pointlabels

                    points = data[_pointlabels]
                    # implementing mapper-assignment
                    if self._mapper.cluster._method == 'single':
                        d = np.min([np.linalg.norm(x-b) for b in points])
                    if self._mapper.cluster._method == 'average':
                        d = np.mean([np.linalg.norm(x-b) for b in points])
                    if self._mapper.cluster._method == 'complete':
                        d = np.max([np.linalg.norm(x-b) for b in points])

                    if d < old_d:
                        toreturn = supernode
                        old_d = d
            assert toreturn, "warning: mistake in assign()"
            return toreturn
        if fx >= max_fx:
            for nodeid in self.partition:
                supernode = self.partition[nodeid]
                if float_eq(supernode._fiber._filter_maxima, max_fx):

                    if leave_one_out:
                        _pointlabels = supernode._pointlabels[supernode._pointlabels != index]
                        if not len(_pointlabels):
                            return supernode
                    else:
                        _pointlabels = supernode._pointlabels

                    points = data[_pointlabels]
                    # implementing mapper-assignment
                    if self._mapper.cluster._method == 'single':
                        d = np.min([np.linalg.norm(x-b) for b in points])
                    if self._mapper.cluster._method == 'average':
                        d = np.mean([np.linalg.norm(x-b) for b in points])
                    if self._mapper.cluster._method == 'complete':
                        d = np.max([np.linalg.norm(x-b) for b in points])

                    if d < old_d:
                        toreturn = supernode
                        old_d = d
            if not toreturn:
                import pdb
                pdb.set_trace()
            assert toreturn, "warning: mistake in assign()"
            return toreturn

    def _predict(self, x, leave_one_out=False, index=None, verbose=0):
        """Helper function called by predict()

        Args:
            x (np.ndarray): one dimensional ndarray representing one data point
        """
        fx = self._mapper.filter.for_assignment_only(x, self._mapper.data)
        if verbose:
            print("f(x) = {}".format(fx))
        supernode = self._assign(x, fx, leave_one_out, index)
        assert supernode, "Error! could not assign x to a corresponding supernode!"
        if verbose:
            print("x assigned to node ", supernode._node._id)
            print("the node {} belongs to a fiber with filter values in ({}, {})".format(supernode._node._id,
                                                                                         supernode._min,
                                                                                         supernode._max))
            print("majority class of node {} is {}".format(supernode._node._id,
                                                           supernode._majorityvote))
        if supernode._pure or not supernode._intervals:
            if verbose:
                print("node is pure! prediction is majorityvote = {}".format(supernode._majorityvote))
            return supernode._majorityvote
        if verbose:
            print("Node is not pure: checking inverted intervals for which the prediction"
                  " would be {}".format(supernode._minorityvote))

        for I in supernode._intervals:
            if verbose:
                print("I = ({}, {})".format(I.a, I.b))
            if fx in I:
                if verbose:
                    print("since fx = {} is in {}<{}<{},"
                          " we return the minorityvote {}".format(fx, I.a, fx, I.b,
                                                                  supernode._minorityvote))
                return supernode._minorityvote
        return supernode._majorityvote

    def predict(self, X, leave_one_out=False, verbose=False):
        predictions = []
        for i, x in enumerate(X):
            predictions.append(self._predict(x, leave_one_out, i, verbose))
            if verbose:
                print("predicting: ", predictions[-1])
        return predictions

    def leave_one_out(self, X, verbose=False):
        return self.predict(X, leave_one_out=True, verbose=verbose)

    def fit(self, verbose=1):
        self._partition = self._disambiguate(verbose)
        self._applyscorefunction(verbose)
        self._computeintervals(verbose)
        self._intervals_to_flip(verbose)
        return self

    def plot_majority_votes(self, pos=None, edge_labels=False, node_labels=False):
        supernode_color = []
        supernode_size = []
        for fiber in self._mapper.cover:
            for node in fiber:
                try:
                    supernode = self.partition[node._id]
                    supernode_color.append(supernode._majorityvote)
                    supernode_size.append(supernode._size)
                except KeyError:
                    self._graph.remove_node(node._id)

        pos = nx.spring_layout(self._graph, iterations=800)
        nx.draw(self._graph, pos=pos,
                node_color=supernode_color, node_size=supernode_size)
        if edge_labels:
            nx.draw_networkx_edge_labels(self._graph, pos=pos,
                                         edge_labels=self._weights)
        if node_labels:
            nx.draw_networkx_labels(self._graph, pos, font_size=8, font_weight='bold', font_color='k')
        plt.show()

    def count(self):
        howmany = 0
        for nodeid in self.partition:
            supernode = self.partition[nodeid]
            howmany += len(supernode._pointlabels)
        print("number of points in the disambiguated mapper: ", howmany)
        print("number of datapoints: ", np.alen(self._mapper.data))
