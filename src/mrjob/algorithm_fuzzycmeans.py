import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJob_fuzzycmeans(MRJob):
    """Single iteration of fuzzy c-means via MRJob. Return the same number of the centroids as the previous iteration.

    Parameters available in all map and reduce functions
    ----------
    centroids : string
                Path to the file where centroids from previous iterations are stored.

    weight : string representing a float
             Weighting parameter for fuzzy c-means algorithm.

    """

    def configure_args(self):
        super(MRJob_fuzzycmeans, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_file_arg('--centroids')
        self.add_passthru_arg('--weight', type=float, help='Weighting exponent (m)')

    def intermediate_membership_matrix(self, _, line):
        """Update one line of the membership matrix.

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        j, (up, u) : int, (list, list)
                     The key is the index of the centroid; the first element of the value
                     is one addend of the numerator in the update equation of the centroids;
                     the second one is one addend of the denominator. Both of them are lists
                     that may be converted to ndarray of shape (1, n_columns).

        """

        l = line.split(",")
        if len(l) == 1:
            return
        point = np.array([float(x) for x in l[1:-1]])

        # Read centroids from file
        with open(self.options.centroids, 'rb') as fcentroids:
            centroids = pickle.load(fcentroids)

        # Weighting exponent (m)
        m = float(self.options.weight)
        # distances
        d_list = [np.linalg.norm(point - c) for c in centroids]
        d_list = [c if c!=0 else 10*(-10) for c in d_list]

        # Element (i,j) of membership matrix
        for j in range(len(d_list)):
            u = [(d_list[j]/d)**(2/(m-1)) for d in d_list]
            u = ( sum(u) )**(-1)
            u = u ** m
            up = u * point

            yield j, [up.tolist(), np.asscalar(u)]

    def update_centroids(self, cluster, up_u):
        """Update one centroid.

        Parameters
        ----------
        cluster : int
                  Index of the centroid.

        up_u : iterator
               Iterator of tuples, obtained with the map function.
               All tuples have the same key and it is equal to cluster.

        Returns
        -------
        cluster, centroid : int, list
                            The key is the index of the centroid; the value is a list, representing a centroid,
                            that may be converted to ndarray of shape (1, n_columns).

        """

        # up_sum numerator and u_sum denominator of update formula for c
        up_sum, u_sum = [], 0
        for k in up_u:
            up_sum.append(k[0])
            u_sum += k[1]
        up_sum = np.sum(np.array(up_sum), axis=0)

        c = up_sum / u_sum

        yield cluster, c.tolist()

    def steps(self):
        return [MRStep(mapper=self.intermediate_membership_matrix, reducer=self.update_centroids)]


if __name__ == '__main__':
    MRJob_fuzzycmeans.run()
