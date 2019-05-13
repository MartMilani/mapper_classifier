"""Author: martino milani
martino.milani94@gmail.com

Test routine for the mapper module
"""
import numpy as np
import lmapper as lm
from lmapper.filter import Eccentricity
from lmapper.cover import BalancedCover
from lmapper.cluster import Linkage
import predmap as mapp
from lmapper.cutoff import FirstGap


def test(x, y):
    """Basic usage"""
    # instantiate a Mapper object
    filter = Eccentricity(exponent=2, metric='correlation')
    cover = BalancedCover(nintervals=4,
                          overlap=0.49)
    cluster = Linkage(method='average',
                      metric='correlation',
                      cutoff=FirstGap(0.01))
    mapper = lm.Mapper(data=x,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit(skeleton_only=False)
    
    print("dimension = ", mapper.complex._dimension)

    predictor = mapp.BinaryClassifier(mapper=mapper,
                                      response_values=y,
                                      _lambda=0.015,
                                      a=0.5,
                                      beta=1)
    predictor.fit()
    # --------------------------
    # predictor.plot_majority_votes()
    return predictor.leave_one_out(x)


def main():
    import pandas as pd
    data = pd.read_csv('../../lmapper/lmapper/datasets/wisconsinbreastcancer.csv')
    x_temp = data[data.columns[2:-1]].values
    y = data[data.columns[1]].values
    y_ = np.asarray([0 if x == 'M' else 1 for x in y])
    x = np.empty(x_temp.shape, dtype='float')
    for i in range(x_temp.shape[0]):
        for j in range(x_temp.shape[1]):
            x[i, j] = x_temp[i, j]
    print(x.shape)
    pred = test(x, y_)
    print('ACCURACY: ', np.mean(pred == y_))

    return 0


if __name__ == '__main__':
    main()
