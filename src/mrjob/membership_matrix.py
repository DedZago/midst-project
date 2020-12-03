import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJob_membership_matrix(MRJob):
    """This job calculates the elements of the membership matrix U using its update equation.

    Parameters available in all map and reduce functions
    ----------
    centroids : string
                Path to the file where centroids are stored.

    weight : string representing a float
             Weighting parameter for fuzzy c-means algorithm.

    Notes
    -----
    The reducer does not create the membership matrix by joining the lines obtained with the map function
    because it may not be possible to save the whole matrix in memory. Doing so, MrJob return an iterator.

    """


    def configure_args(self):
        super(MRJob_membership_matrix, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_file_arg('--centroids')
        self.add_passthru_arg('--weight', type=float, help='Weighting exponent (m)')

    def intermediate_membership_matrix(self, _, line):
        """Calculate one line of the membership matrix.

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        i, u_line : int, list
                    The key is the row index; the value is list of numbers, representing the i-th row of U.

        """

        l = line.split(",")
        if len(l) == 1:
            return
        point = np.array([float(x) for x in l[1:-1]])

        # Read centroids from file
        with open(self.options.centroids, 'rb') as fcentroids:
            centroids = pickle.load(fcentroids)

        # Weighting exponent (m)
        m = self.options.weight

        # Distances are stored in a list
        d_list = [np.linalg.norm(point - c) for c in centroids]
        d_list = [c if c!=0 else 10*(-10) for c in d_list]

        # Element (i,j) of membership matrix, i fixed
        u_line = []
        for j in range(len(d_list)):
            u = [(d_list[j]/d)**(2/(m-1)) for d in d_list]
            u = ( sum(u) )**(-1)
            u_line.append(np.asscalar(u))

        yield l[0], u_line


    def identity(self, i, u_line):
        """Identity function.

        Parameters
        ----------
        i : string representing an int
            Row index
        u_line: iterator
                Iterator containing one list of numbers, representing the i-th row of U.

        Returns
        -------
        i, u_line : int, list
                    The key is the row index; the value is list of numbers, representing the i-th row of U.

        """
        yield int(i), [int(i)] + next(u_line)

    def steps(self):
        return [MRStep(mapper=self.intermediate_membership_matrix, reducer=self.identity)]


if __name__ == '__main__':
    MRJob_membership_matrix.run()
