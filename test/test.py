"""Author: maritno milani
m.milani@l2f.ch

Test routine for the shapegraph module
"""
import numpy as np
import lmapper as lm
from lmapper.filter import Projection
from lmapper.cover import BalancedCover
from lmapper.cluster import Linkage
import predmap as mapp


def test(x, y):
    """Basic usage"""
    # instantiate a ShapeGraph object

    # filter = Eccentricity(exponent=2)
    filter = Projection(ax=2)
    cover = BalancedCover(nintervals=20,
                          overlap=0.4)
    cluster = Linkage(method='single')
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(verbose=1)
    predictor = mapp.BinaryClassifier(mapper=mapper,
                                      response_values=y,
                                      _lambda=0.4)
    predictor.fit(verbose=1)
    return predictor.predict(x[[1, 500, 1000]])


def main():
    from numpy import genfromtxt
    x = genfromtxt('/Users/martinomilani/Documents/III_semester/PACS/project/synthetic_dataset/synthetic.csv', delimiter=',')
    x = x[1:]  # eliminating the first row of nans
    y = np.asarray([row[3] for row in x])
    x = np.asarray([row[0:3] for row in x])
    test(x, y)
    return 0


if __name__ == '__main__':
    main()
