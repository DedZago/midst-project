from src.file_io import extract_mrjob

def fuzzy_cmeans(fdata, fcentroids, m, **kwargs):
    """Wrapper to call one iteration of fuzzy c-means via MRJob.

    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    fcentroids : string
                 Path to the file where centroids from previous iterations are stored.

    m : float or string representing a float
        Weighting parameter for fuzzy c-means algorithm.

    runner : string
             Argument passed via the json settings, specifying if Mrjob must work in parallel ("local")
             or sequentially ("inline").

    Returns
    -------
    centroids : ndarray of shape (k, n_columns)
                Matrix with centroids updated by one iteration of fuzzy c-means as rows.
                
    """

    from src.mrjob.algorithm_fuzzycmeans import MRJob_fuzzycmeans

    update_centroids_job = MRJob_fuzzycmeans(args=[fdata, "--runner", kwargs["runner"], "--centroids", fcentroids, "--weight", str(m), "--no-bootstrap-mrjob"])
    with update_centroids_job.make_runner() as update_centroids_runner:
        update_centroids_runner.run()
        centroids = extract_mrjob(update_centroids_job, update_centroids_runner)
    return centroids


def kmeans(fdata, fcentroids, **kwargs):
    """Wrapper to call one iteration of k-means via MRJob.

    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    fcentroids : string
                 Path to the file where centroids from previous iterations are stored.

    runner : string
             If runner="local", MrJob works in parallel; if runner="inline", MrJob works sequentially.

    Returns
    -------
    centroids : ndarray of shape (k, n_columns)
                Matrix with centroids updated by one iteration of k-means as rows.

    """

    from src.mrjob.algorithm_kmeans import MRJob_kmeans

    update_centroids_job = MRJob_kmeans(args=[fdata, "--runner", kwargs["runner"], "--centroids", fcentroids, "--no-bootstrap-mrjob"])
    with update_centroids_job.make_runner() as update_centroids_runner:
        update_centroids_runner.run()
        centroids = extract_mrjob(update_centroids_job, update_centroids_runner)

    return centroids
