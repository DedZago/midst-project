import csv
import numpy as np
import os
import random

from src.file_io import centroids_to_disk, extract_mrjob


def init_plusplus(fdata, k, p, **kwargs):
    """Plusplus initialization via MRJob.

    The initialization is performed by randomly choosing the first centroid from the data file
    and then applying the plusplus algorithm via MRJob.

    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    k : int or string (representing an integer)
        Argument passed via the json settings, specifying the number of centroids to initialize.

    p : float or string (representing a float)
        Argument passed via the json settings, specifying the value of the parameter that determines
        how much the centroids are spread through the space.

    Returns
    -------
    centroids : ndarray of shape (k, n_columns)
                Matrix with selected centroids as rows.

    Notes
    -----
    This function saves the first selected centroid as a temporary file "tmp_centroids", in order to
    feed it to the MRJob worker; the file is deleted upon successful initialization of all centroids.

    """

    from src.mrjob.init_plusplus import MRJob_plusplus

    with open(fdata) as f:
        lines = sum(1 for line in f)
        line_number = random.randrange(lines)

    with open(fdata) as f:
        reader = csv.reader(f)
        centroids = np.array(next(row for row_number, row in enumerate(reader)
                                   if row_number == line_number))[1:-1]
        centroids = np.array([float(i) for i in centroids])

    # create set of selected centroids
    centroids_to_disk(centroids, "tmp_centroids")

    for i in range(1, int(k)):
        init_centroids_job = MRJob_plusplus(args=[fdata, "--runner", kwargs["runner"], "--centroids", "tmp_centroids", "--p", str(p), "--no-bootstrap-mrjob"])
        with init_centroids_job.make_runner() as init_centroids_runner:
            init_centroids_runner.run()
            centroids = np.vstack((centroids, extract_mrjob(init_centroids_job, init_centroids_runner, "key")))
            # update set of selected centroids
            centroids_to_disk(centroids, "tmp_centroids")

    # delete temporary file created for plusplus initialization
    os.remove("tmp_centroids")

    return centroids


def init_step(fdata, k, **kwargs):
    """Step initialization via MRJob.

    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    k : int or string representing an integer
        Argument passed via the json settings, specifying the number of centroids to initialize.

    Returns
    -------
    centroids : ndarray of shape (k, n_columns)
                Matrix with selected centroids as rows.
    """

    from src.mrjob.init_step import MRJob_step

    init_centroids_job = MRJob_step(args=[fdata, "--runner", kwargs["runner"], "--k", str(k), "--no-bootstrap-mrjob"])
    with init_centroids_job.make_runner() as init_centroids_runner:
        init_centroids_runner.run()
        centroids = extract_mrjob(init_centroids_job, init_centroids_runner)

    return centroids


def init_random(fdata, k, **kwargs):
    """Random initialization via MRJob.

    This function chooses k random data points by iterating twice over the data set.
    With the first iteration the number of rows is counted, with the second iteration the
    k centroids are selected.

    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    k : int or string representing an integer
        Argument passed via the json settings, specifying the number of centroids to initialize.

    Returns
    -------
    centroids : ndarray of shape (k, n_columns)
                Matrix with selected centroids as rows.
    """

    from src.mrjob.init_random import MRJob_random

    with open(fdata) as f:
        lines = sum(1 for line in f)
        line_number = random.randrange(lines)

    with open(fdata) as f:
        reader = csv.reader(f)
        centroids = np.array(next(row for row_number, row in enumerate(reader)
                                   if row_number == line_number))[1:-1]
        centroids = np.array([float(i) for i in centroids])

    # create set of selected centroids
    centroids_to_disk(centroids, "tmp_centroids")

    for i in range(1, int(k)):
        init_centroids_job = MRJob_random(args=[fdata, "--runner", kwargs["runner"], "--centroids", "tmp_centroids", "--no-bootstrap-mrjob"])
        with init_centroids_job.make_runner() as init_centroids_runner:
            init_centroids_runner.run()
            centroids = np.vstack((centroids, extract_mrjob(init_centroids_job, init_centroids_runner, "key")))
            # update set of selected centroids
            centroids_to_disk(centroids, "tmp_centroids")

    # delete temporary file created for random initialization
    os.remove("tmp_centroids")

    return centroids
