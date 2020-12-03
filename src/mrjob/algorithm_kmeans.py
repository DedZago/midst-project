import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJob_kmeans(MRJob):
    """Single iteration of k-means via MRJob. Return the same number of the centroids as the previous iteration.

    Parameters available in all map and reduce functions
    ----------
    centroids : string
                Path to the file where centroids from previous iterations are stored.

    """

    def configure_args(self):
        super(MRJob_kmeans, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_file_arg('--centroids')

    def assign_cluster(self, _, line):
        """Assign a point to the nearest centroid.

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        cluster, point : int, list
                         The key is the index of the centroid; the value is a list, representing the point,
                         that may be converted to ndarray of shape (1, n_columns).

        """

        l = line.split(",")
        if len(l) == 1:
            return

        point = np.array([float(x) for x in l[1:-1]])

        # Read centroids from file
        with open(self.options.centroids, 'rb') as f:
            centroids = pickle.load(f)

        distances = [np.linalg.norm(point - c) for c in centroids]
        cluster = np.argmin(distances)

        yield int(cluster), point.tolist()

    def compute_average(self, cluster, points):
        """Update one centroid.

        Parameters
        ----------
        cluster : int
                  Index of the centroid.

        points : iterator
                 Iterator of the points assigned to the same cluster. Each point is a list,
                 that may be converted to ndarray of shape (1, n_columns).

        Returns
        -------
        cluster, centroid : int, list
                            The key is the index of the centroid; the value is a list, representing a centroid,
                            that may be converted to ndarray of shape (1, n_columns).

        """

        s = np.array(next(points))
        n = 1
        for x in points:
            s += x
            n += 1

        yield cluster, (s / n).tolist()

    def steps(self):
        return [MRStep(mapper=self.assign_cluster, reducer=self.compute_average)]


if __name__ == '__main__':
    MRJob_kmeans.run()
