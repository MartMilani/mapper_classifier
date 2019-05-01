"""Author: Martino Milani
martino.milani94@gmail.com

Test routine for the predmap module
"""
import numpy as np

import lmapper as lm
from lmapper.filter import Projection
from lmapper.cover import UniformCover
from lmapper.cluster import Linkage
import predmap as mapp
from lmapper.cutoff import FirstGap


def test(x, y):
    """Basic usage"""

    print(x.shape)
    # instantiate a Mapper object

    filter = Projection(ax=2)
    cover = UniformCover(nintervals=25,
                         overlap=0.4)
    cluster = Linkage(method='single',
                      metric='euclidean',
                      cutoff=FirstGap(0.05))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=True)
    print("dimension = ", mapper.complex._dimension)

    predictor = mapp.BinaryClassifier(mapper=mapper,
                                      response_values=y,
                                      _lambda=0.15,
                                      a=0.5,
                                      beta=2)
    predictor.fit()  # .plot_majority_votes()

    return predictor.leave_one_out(x)


def main():
    from numpy import genfromtxt
    x = genfromtxt('../../lmapper/lmapper/datasets/synthetic.csv',
                   delimiter=',')

    # preprocessing of data
    # eliminating the first row of nans
    x = x[1:]
    # separating features and labels
    y = np.asarray([row[3] for row in x])
    x = np.asarray([row[0:3] for row in x])

    pred = test(x, y)

    print('ACCURACY: ', np.mean(pred == y))

    return 0


if __name__ == '__main__':
    main()
