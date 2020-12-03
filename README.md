# Distributed clustering via MapReduce

In this work we propose an implementation of various clustering algorithms in a distributed and parallel local *MapReduce* environment, via the [mrjob](https://mrjob.readthedocs.io/en/latest/) package.

## Getting Started

These instructions will let you replicate the tests on your local machine.

### Prerequisites
The programming language used is [Python 3.6.8](https://www.python.org/downloads/).

The following Python packages are required:
* [numpy](https://numpy.org/)
* [mrjob](https://mrjob.readthedocs.io/en/latest/)

The command-line package installer used is [pip](https://pypi.org/project/pip/).

### Installing
Python can be downloaded and installed from the [official website](https://www.python.org/downloads/).
You can check your Python version from the command prompt as follows:

```python3 --version```

If the pip installer needs updating, it can be done as follows:

```python3 -m pip install --upgrade pip``` (Windows)

```sudo pip3 install --upgrade pip``` (Linux)

The required packages can be installed from the command prompt as follows:

```pip3 install -r requirements.txt```

**Note:** based on your machine, you may need to use `python` and `pip`, instead of `python3` and `pip3` from the command prompt

## Contents
The main module of the project is `main.py`, which allows to perform one or more complete processes of *cluster analysis*. One process, understood as the series of passages necessary to divide a dataset in *k* clusters, is determinated by a combination of the following parameters:
* *number of clusters*, or equivalently *number of centroids*
* *initialization*, used for selecting the starting centroids
* *algorithm*, used for updating the centroids
* *maximum number of iterations* of the algorithm
* *stopping criterion*, that determines when the algorithm reaches the convergence
* *threshold of the stopping criterion*
* *parameter of  the ++ initialization*
* *fuzzy weighting exponent*, that is a parameter of fuzzy c-means algorithm

In fact, `main.py` is a wrapper used to call in sequence all the functions necessary to complete a process: based on the parameters received in input through `settings.json`, it imports different modules from the folder [src](https://github.com/DedZago/midst/tree/master/src) and performs different initializations, algorithms or stopping criterions.

The repository also includes the following folders:
* *[centroids](https://github.com/DedZago/midst/tree/master/centroids)* \
	It contains the centroids files created by `main.py`; for each process, `main.py` creates a pickle file containing the centroids and stores it in this folder. The file name is built as follows:

	`<id>_centroids_<algorithm>_<initialization>_<stop_crit>_<k>`,
    
	where each string in `<...>` corresponds to a different parameter previously exposed.

* *[data](https://github.com/DedZago/midst/blob/local/data)* \
	It contains the datasets used to test the various implementations. Each dataset has the features of interest in columns $2$ to $p-1$, where p number of columns.
    
    **Important**: it is always assumed that the first column contains a row identifier and the last one contains the correct class label. If this is not desired, the only scripts that require change are `initialization.py` and those inside the `src/mrjob` folder. The required modification is
    
    `point = np.array([float(x) for x in l[1:-1]])` ==> `point = np.array([float(x) for x in l)`
    
* *[src](https://github.com/DedZago/midst/tree/master/src)* \
	It contains the modules imported by `main.py`:
	* `file_io.py`: utility functions for file input-output
	* `algorithm.py`: wrappers calling algorithms
	* `initialization.py`: wrappers calling initializations
	* `stopping_criterion.py`: implementation of the stopping criterion

* *[src/mrjob](https://github.com/DedZago/midst/tree/master/src/mrjob)* \
	It contains the modules that implements the *MapReduce jobs* via *MrJob*; they are imported by the wrappers in the parent folder.   

## Running the tests
Before performing a test, you need to modify `setting.json` in order to select the desired combinations of parameters. In addition to the ones previously exposed, you may also specify two additionals parameters, that do not influence the result of the processes:
* *parallel*: if *true*, *mrjob* works in parallel
* *logging*: if *true*, information about the processes completed are  stored in `log.txt`

The keys associated with the various parameters are shown in the table below: if *type*=*list*, the value must be a list of elements compatible with *possible values*; otherwise, if *type*=*element*, the value must an element compatible with *possible values*.
| key                          | type    | possible values              |
|------------------------------|---------|------------------------------|
| id                           | element | *string*                     |
| centroids                    | list    | *integer*, or "a-b" with a<b |
| initialization               | list    | "random", "step", "fuzzy"    |
| algorithm                    | list    | "kmeans", "fuzzy-cmeans"     |
| max_iterations               | element | *integer*                    |
| stopping_criterion           | list    | "biggest-diff"               |
| stopping_criterion_threshold | element | *float* (>0)                 |
| plusplus_parameter           | element | *float* (>1)                 |
| fuzzy_weighting_exponent     | list    | *float* (&#8805;1)           |
| parallel                     | element | *boolean*                    |
| logging                      | element | *boolean*                    |

**notes:**
* In *centroids*, "a-b" is the same as writing all numbers between a and b (including the end values).
* In *centroids* and *fuzzy_weighting_exponent*, duplicate numbers are removed and the list is sorted in ascending order.
* *max_iterations*, *parallel* and *logging* are the only parameters with a default value, respectively 50, *false* and *false*.
* If `setting.json` does not contain values for all parameters without a default value or some values are incorrect, a message is shown and the processing does not start.

Finally, you can launch the main module from the command prompt as follows:

```python3 main.py data/<dataset>```,

where `<dataset>` is the dataset on which to apply the cluster analysis.

## Other modules
The pickle files created by `main.py` are not human-readable, therefore a module that allows to read those files is needed.
You can launch the module from the command prompt as follow:

```python3 print_file.py <file_to_print>```,

where `<file_to_print>` is the pickle file to print.

For each process where the algorithm is *fuzzy c-means*, `main.py` does not create a file containing the *membership matrix* since it may be very large; however, it is possible to create it using  `centroids_to_mb.py`, that is a wrapper that imports a module  implementing a *MapReduce job* via *MrJob*.

You can launch the module from the command prompt as follows:

```python3 centroids_to_mb.py <fdata> <fcentroids>```,

where `<fdata>` is the path to the .csv file containing the dataset used to create `<fcentroids>`, `<fcentroids>` is the name of the file containing the centroids of interest.

`centroids_to_mb.py` works with the default name of the centroid file only since it extracts the weighting exponent from it, moreover
the `<fcentroids>` must be stored in the *[centroids](https://github.com/DedZago/midst/tree/master/centroids)* folder.

## Authors

* **Daniele Zago** - [DedZago](https://github.com/DedZago)
* **Giovanni Toto** - [giovannitoto](https://github.com/giovannitoto)
