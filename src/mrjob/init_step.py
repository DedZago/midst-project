import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJob_step(MRJob):
    """Implementation of step initialization via MrJob.

    Parameters available in all map and reduce functions
    ----------
    k : string (representing a float)
        Number of centroids to initialize.

    """

    def configure_args(self):
        super(MRJob_step, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_passthru_arg('--k', type=int, help='Number of clusters')

    def get_points(self, _, line):
        """Convert the the line of the .csv file to a list, that may be converted
        to ndarray of shape (1, n_columns).

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        _, point : _, list
                   The key is not specified in order to aggregate all couples in the same reducer;
                   the value is a list, representing the point, that may be converted
                   to ndarray of shape (1, n_columns).

        """

        point = line.split(",")
        if len(point) == 1:
            return

        yield None, [float(x) for x in point[1:-1]]

    def get_centroids(self, _, points):
        """Obtain k centroids using a step.

        Parameters
        ----------
        points : Iterator of lists, representing the points of the dataset, obtained with the map function.

        Returns
        -------
        _, centroid : _, list
                      The key is not specified since is not necessary to a index to a centroid.
                      the value is a list, representing a centroid, that may be converted
                      to ndarray of shape (1, n_columns).

        """
        # Get min value for each feature
        minp = maxp = np.array(next(points), dtype=float)
        for x in points:
            minp = np.minimum(minp, x)
            maxp = np.maximum(maxp, x)
        # Get step
        k = self.options.k
        step = (maxp - minp) / k
        # Get k centroids
        for i in range(k):
            c = minp + step * i
            yield None, c.tolist()

    def steps(self):
        return [MRStep(mapper=self.get_points, reducer=self.get_centroids)]

if __name__ == '__main__':
    MRJob_step.run()
