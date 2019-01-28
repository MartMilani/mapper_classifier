import numpy as np
from scipy.spatial.distance import pdist, squareform


class Interval():
    """Helper class to ease writing 'if a in I' in SuperNode.predict()"""
    def __init__(self, center, a, b):
        self.a = a
        self.b = b

    def __contains__(self, item):
        return item < self.b and item > self.a


class SuperNode():
    """Class containing all the data structures and implementing all the base routines of
    the predictive Mapper algorithm.

    The aim of this class is to extend the class shapeegraph.Node with attributes and
    methods necessary to implement the predictive Mapper algorithm.
    Each object of this class contains a reference to one and only one shapegraph.Node().


    Attributes:
        _node (shapegraph.Node): the node in the shapegraph that each object refer to.
        _fiber (shapegraph.Fiber): the fiber in the shapegraph that this Supernode belongs
            to.
        _shapegraph (shapegraph.Shapegraph): the shapegraph.Shapegraph this SuperNode
            belongs to
        _pointlabels (np.ndarray): subset of _node._pointlabels containing the point
            labels of the disambiguated node i.e. the points belonging to self._node and
            not belonging to any other node in the shapegraph. Remember that these point
            labels are the integer indexes of the shapegraph._data matrix
        _score (float): score S that will determine how many intervals will be switched
        _responsevalues (np.ndarray): array containing the response values corresponding
            to _pointlabels
        _min (float): minimum value of the filter function
        _max (float): maximum value of the filter function
        _size (float): len(self._pointlabels)
        _densitydetectors (np.ndarray): array of delta(x)=max_{k}{1/k*g_k(x)*score}
            corresponding to every point x in self._pointlabels
        _intervals (list): list of Intervals for which the prediction is
            self._minorityvote, after having appied Disabuigation, scoring, ranking and
            updating_predictions.
        _distancematrix (np.ndarray): distance matrix self.size X self.size, containing
            the distances of the response values corresponding to self._pointlabels
        _neighboursmatrix (np.ndarray): matrix self.size  X self.size, where each
            i-th column is a copy of the array self._pointlabels, sorted in a way that the
            element (j, i) is the label of the j-th closest point to the point i. This
            matrix allows the construction of the function g_k(x). Pay attention that
            the element (0, i) = i, since d(i,i) = 0 < d(i,j) for every j.
        _majorityvote (float): simply the mayority vote of the SuperNode
        _minorityvote (float): simply the opposite of _majorityvote
        _ones (float): how many _pointlabels belongs to the class 1
        _zeros (float): how many _pointlabels belongs to the class 0
        _ks (np.ndarray): array of length self.size, containing the argmax_{k}{1/k*g_k(x)}
            corresponding to each point x in self._pointlabels
        _intervalscores (np.ndarray):
            _intervalscores = [4, 6, 8, ...] means that
            densitydetector[4] = delta(shapegraph._data[pointlabels[4]]) >
            densitydetector[6] = delta(shapegraph._data[pointlabels[6]]) >
            densitydetector[8] = delta(shapegraph._data[pointlabels[8]]) >
            ...
        _pure (bool): True if node is pure i.e. it doesn't contain any switched interval
        myfiltervalues (np.ndarray): filtervalues corresponding to _pointlabels
        _extrapointlabels (np.ndarray): used in the disambiguation step, contains
            the pointlabels of points belonging to intersections that have been added
            to this supernode as a consequence of the disambiguation step.
        """

    def _finduniquelabels(self, y):
        """Function that by using the informations contained in self.shapegraph.complex._intersection_dict,
        it removes form self._node._pointlabels all the points contained in at least
        one other node
        """
        # initializing the pointlabels with the ones of the underlying node
        self._pointlabels = self._node._labels
        # subtracting any intersections from the labels
        intersectiondict = self._shapegraph.complex._intersection_dict
        for simplex in intersectiondict.keys():  # type(simplex) == tuple
            # extracting points that belong to self._node
            if self._node._id in simplex:
                self._pointlabels = np.setdiff1d(self._pointlabels, intersectiondict[simplex])
        self._size = len(self._pointlabels)  # this value is only temporary! will be changed
        self._myonlyones = sum(y[self._pointlabels])
        self._myonlyzeros = self._size - self._myonlyones

    def _findresponsevalues(self, y):
        """Helper function called by __init__()

        Args:
            y (np.ndarray): array containing the response values of the whole training set
                used to construct the shapegraph"""
        self._responsevalues = y[self._pointlabels]
        self.myfiltervalues = self._shapegraph.filter_values[self._pointlabels]
        self._min = min(self.myfiltervalues)
        self._max = max(self.myfiltervalues)

    def _setmajorityvote(self):
        """Helper function called by __init__()"""

        self._ones = int(np.sum(self._responsevalues))
        self._zeros = int(self._size - self._ones)
        self._extraones = self._ones-self._myonlyones
        self._extrazeros = self._zeros-self._myonlyzeros
        self._majorityvote = (self._ones > self._zeros)
        self._minorityvote = not self._majorityvote
        if self._ones == self._size or self._zeros == self._size:
            self._pure = True

    def __init__(self, node, fiber, shapegraph):
        """
        Args:
            node (sg.Node): node
            fiber (sg.Fiber): fiber that contains node
            shapegraph (sg.ShapeGraph): shapegraph object
            y (np.ndarray): one dimensional array containing the labels of all the data
                points contained in shapegraph.data"""
        self._node = node
        self._fiber = fiber
        self._shapegraph = shapegraph
        self._pointlabels = None
        self._score = None
        self._responsevalues = None
        self._min = None
        self._max = None
        self._size = None
        self._densitydetectors = None
        self._intervals = []
        self._distancematrix = None
        self._neighboursmatrix = None
        self._majorityvote = None
        self._minorityvote = None
        self._ones = None
        self._zeros = None
        self._ks = None
        self._intervalscores = None
        self._pure = False
        self.myfiltervalues = None
        self._extrapointlabels = []
        self._extraones = None
        self._extrazeros = None
        self._myonlyones = None
        self._myonlyzeros = None

    def _applyscorefunction(self, a):
        self._score = a * abs(self._myonlyones - self._myonlyzeros) + (1-a) * abs(self._extraones - self._extrazeros)

    def _computeintervals(self, beta=1, verbose=1):
        """This function
            - computes the values of g(k, x) for every k, x
            - computes the corresponding max_{k}{g} and argmax_{k}{g}

            Args:
                beta (float): exponent of the factor (1/k)^beta used in the calculation of
                the density detector. The higher beta, the less farther neighbours
                influence the density detector
        """
        flatdistances = pdist(self.myfiltervalues.reshape(self._size, 1))
        self._distancematrix = squareform(flatdistances)
        self._neighboursmatrix = np.argsort(self._distancematrix, axis=1)
        #
        #                       | 0   x_{01} x_{02} ...      |
        #                       | 1   x_{11} x_{12} ...      |
        #  _neighboursmatrix =  | 2   x_{21} x_{22} ...      |
        #                       | 3   x_{31} x_{32} ...      |
        #                       | ..  ..      ..    ...      |
        #                       | ..  ..      ..    ...      |
        #
        # where x_{ij} is the index in self._pointlabels of the j-th closest filter value
        # to the i-th filter value

        full_index = np.asarray([x for x in range(self._size)])
        if self._majorityvote:
            cols = self._zeros
            col_indices = full_index[self._responsevalues == 0]
        else:
            cols = self._ones
            col_indices = full_index[self._responsevalues == 1]
        gk = np.zeros((cols, self._size))
        #
        #        | 1   x_{01} x_{02} ...      |
        #        | 1   x_{11} x_{12} ...      |
        #  gk =  | 1   x_{21} x_{22} ...      |
        #        | 1   x_{31} x_{32} ...      |
        #        | ..  ..      ..    ...      |
        #        | ..  ..      ..    ...      |
        #
        # where x_{ij} is the value of gk_{x_i}(j), where
        #
        # sameclass = how many points with the same label of x_i;
        # oppositeclass = how many points with the opposite label of x_i
        # gk_{x_i}(j) = ( sameclass / j)^beta * max(sameclass - oppositeclass, 0)

        # there are in a j-neighbourhood of x_i
        for i, index in enumerate(col_indices):
            sameclass = 0
            oppositeclass = 0
            for j in range(self._size):
                if self._responsevalues[j] == self._minorityvote:
                    new_neighbours_class = self._responsevalues[self._neighboursmatrix[index, j]]
                    if new_neighbours_class == self._minorityvote:
                        sameclass += 1
                    else:
                        oppositeclass += 1
                    if j:
                        gk[i, j] = (float(sameclass) / float(j)) * (max(sameclass - oppositeclass, 0) ** beta)
                        # thesis version:
                        # gk[i, j] = (1 / j) * (max(sameclass - oppositeclass, 0) ** beta) * sameclass
                    else:
                        gk[i, 0] = max(sameclass - oppositeclass, 0)
                        # gk[i, 0] should be = 1! use this fact to debug
        self._ks = np.argmax(gk, axis=1)
        self._densitydetectors = [gk[i, k] for i, k in enumerate(self._ks)]  # should be np.max(gk, axis=1)
        # _ks = | k_0 k_1 ... k_N]
        # _densitydetectors = | d_0 d_1 ... d_N|
        # where d_i = gx[i, k_i] is the maximum value of gk in the i-th row
        self._intervalscores = np.argsort(self._densitydetectors)
        # x = intervalscores[i]
        # Interval(x, _densitydetectors[x], _ks[x]) is the i-th interval to change
        if verbose:
            pass

    def _updatepredictions(self, _lambda, verbose):
        shouldswitch = int(np.round(self._score*_lambda))
        if self._minorityvote:
            maxswitch = self._ones
        else:
            maxswitch = self._zeros
        intervalstoswitch = np.round(np.min([shouldswitch, maxswitch]))

        if verbose:
            print('\nNode {}'.format(self._node._id))
            print('Score: {}'.format(self._score))
            print('Number of intervals to switch: {}'.format(intervalstoswitch))

        howmanycollapsed = 0
        for i in range(intervalstoswitch):
            x = self._intervalscores[i]
            k = self._ks[x]
            if not k:  # this means that the interval is collapsed to the only point
                howmanycollapsed += 1
                continue
            neighbours = self._neighboursmatrix[x, :k+1]
            neighbours_filtervalues = self.myfiltervalues[neighbours]
            a = np.min(neighbours_filtervalues)
            b = np.max(neighbours_filtervalues)
            self._intervals.append(Interval(x, a, b))
            if verbose:
                print('Filter interval actually switched: I = ({}, {})'.format(a, b))

        if verbose:
            print('{} intervals are actually collapsed to a single point'.format(howmanycollapsed))
