import numpy as np

def biggest_diff(centroids, new_centroids):
    """Largest euclidean distance between of old centroids and new centroids.

    Parameters
    ----------
    centroids : ndarray of shape (k, n_features)
                Contains old centroids.

    new_centroids : ndarray of shape (k, n_features)
                Contains new centroids.

    Returns
    -------
    max_d : float
            Largest difference between old and new centroids.
    """
    distances = [np.linalg.norm(np.array(c1) - c2) for c1,c2 in zip(centroids,new_centroids)]
    max_d = max(distances)
    return max_d
