Lmapper
-------

This package implements the Mapper algorithm.

Example:

    >>> import lmapper as lm
    >>> import predmap as pmapp
    >>>
    >>> N = 100
    >>> d = 10
    >>> x, y = np.random.rand(size=(N, d)), np.randint(size=(N,))
    >>> mapper = lm.Mapper(data=x,
    >>>                    filter='Projection',
    >>>                    cluster='Linkage',
    >>>                    cover='BalancedCover)
    >>> mapper.fit()
    >>> predictor = pmapp.BinaryClassifier(mapper=mapper,
    >>>                                    response_values=y,
    >>>                                    _lambda=0.015,
    >>>                                    a=0.5,
    >>>                                    beta=2)
    >>> predictor.fit()
    >>> predictor.predict()
