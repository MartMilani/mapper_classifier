from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='predmap',
      version='0.1',
      description=('This is a Python implementation of Francesco Palma\'s classifier'
                   'as described in his master thesis "A Mapper based approach '
                   'for predictive data analysis"'),
      long_description='',
      url='http://github.com/MartMilani/predictive_mapper',
      author='Martino Milani',
      author_email='matrino.milani94@gmail.com',
      license='MIT',
      packages=['predmap'],
      install_requires=[
        'lmapper',
        'numpy',
        'scipy',
        'sklearn',
        'networkx',
        'matplotlib',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
