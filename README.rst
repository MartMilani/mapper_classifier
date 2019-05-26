Lmapper
-------

This package implements a classification algorithm based on Mapper.

Example:

    >>> import lmapper as lm
    >>> import predmap as pm
    >>> from lmapper.datasets import synthetic_dataset
    >>>
    >>>
    >>> x, y = synthetic_dataset()
    >>> mapper = lm.Mapper(data=x,
    >>>                    filter='Projection',
    >>>                    cluster='Linkage',
    >>>                    cover='BalancedCover')
    >>> mapper.fit()
    >>> classifier = mapp.BinaryClassifier(mapper=mapper,
    >>>                                    response_values=y,
    >>>                                    _lambda=0.0,
    >>>                                    a=0.5,
    >>>                                    beta=2)
    >>> classifier.fit()


How to install on Mac OS High Sierra 10.13.6
--------------------------------------------
run the following commands

$ cd <path_to_this_directory>
$ pip install .

this command will read the setup.py file in this directory and install predmap

Check your installation
-----------------------

Run the following tests:

$ python <path-to-this-folder>/test/test_synthetic.py
$ python <path-to-this-folder>/test/test_wisconsin_breast_cancer.py

If no exeption is raised, the installation should be successful
