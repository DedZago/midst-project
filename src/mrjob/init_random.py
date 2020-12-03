import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep


class MRJob_random(MRJob):
    """Modification of the algorithm proposed in Bodoia M., 'MapReduce Algorithms for k-means Clustering',
    in order to allow random initialization.

    Parameters available in all map and reduce functions
    ----------
    centroids : string
                Path to the file where the means in M are stored.

    p : string (representing a float)
        Parameter that determines how much the centroids are spread through the space.

    """

    def configure_args(self):
        super(MRJob_random, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_file_arg('--centroids')

    def map_choice(self, _, line):
        """Checks if point was already selected and assigns probability of being chosen.

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        _, (point, distance) : None, (list, integer)
                   The key is not specified in order to aggregate all couples in the same reducer.
                   The first element of the value is a list, representing the point, that may be converted to ndarray of shape (1, n_columns);
                   The second element is either 1, if the point is yet to be selected, or 0 otherwise.

        """

        l = line.split(",")
        if len(l) == 1:
            return

        point = np.array([float(x) for x in l[1:-1]])

        # Read centroids from file
        with open(self.options.centroids, 'rb') as f:
            centroids = pickle.load(f)

        # Distance between points and nearest mean m
        for c in centroids:
            if (np.array(point) == c).all():
                d = 0
            else:
                d = 1

        yield None, (point.tolist(), d)


    def reduce_choice(self, _, pair):
        """Choose the next mean mx randomly from the set X, where the probability
        of a point x in X being chosen is proportional to 1/n, and add mx to M.

        Parameters
        ----------
        pair : iterator
               Iterator of all tuples obtained with the map function.

        Returns
        -------
        m, xy : list, float
                The key is a list, representing the next mean mx, that may be converted
                to ndarray of shape (1, n_columns); the value is not of interest.

        """

        x = next(pair)
        # Choose point with probability proportional to its distance
        for y in pair:
            if np.random.uniform() <= x[1]/(x[1]+y[1]):
                x = (x[0], x[1] + y[1])
            else:
                x = (y[0], x[1] + y[1])

        yield x


    def steps(self):
        return [MRStep(mapper=self.map_choice, reducer=self.reduce_choice)]

if __name__ == '__main__':
    MRJob_random.run()
