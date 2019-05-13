"""Author: Martino Milani
martino.milani94@gmail.com

Test routine for the predmap module
"""
import numpy as np

import lmapper as lm
from lmapper.filter import Projection
from lmapper.cover import UniformCover
from lmapper.cluster import Linkage
from lmapper.datasets import synthetic_dataset
from lmapper.cutoff import FirstGap
import predmap as mapp


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
                                      _lambda=0.0,
                                      a=0.5,
                                      beta=2)
    predictor.fit()  # .plot_majority_votes()

    return predictor.leave_one_out(x)


def main():
    x, y = synthetic_dataset()
    pred = test(x, y)
    print('ACCURACY: ', np.mean(pred == y))
    return 0


if __name__ == '__main__':
    main()
