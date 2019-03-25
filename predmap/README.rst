predmap
-------

This package implements the Mapper classifier algorithm.

Example:

    >>> import lmapper as lm
    >>> import predmap as pm
    >>>
    >>> N = 100
    >>> d = 10
    >>> x, y = np.random.rand(size=(N, d)), np.randint(size=(N,))
    >>> mapper = lm.Mapper(data=x,
    >>>                    filter='Projection',
    >>>                    cluster='Linkage',
    >>>                    cover='BalancedCover)
    >>> mapper.fit()
    >>> predictor = pm.BinaryClassifier(mapper=mapper,
    >>>                                 response_values=y,
    >>>                                 _lambda=0.015,
    >>>                                 a=0.5,
    >>>                                 beta=2)
    >>> predictor.fit()
    >>> x0 = np.random.rand(size=(1, d))
    >>> predictor.predict(x0)
