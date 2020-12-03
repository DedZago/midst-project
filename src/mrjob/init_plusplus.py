import numpy as np
import pickle

from mrjob.job import MRJob
from mrjob.step import MRStep


class MRJob_plusplus(MRJob):
    """Implementation of the 2-3 steps of the following algorithm, proposed in Bodoia M., 'MapReduce Algorithms for k-means Clustering':

        1. Choose the first mean m0 uniformly at random from the set of points X and add it to the set M.
        2. For each point x in X, compute the distance D(x) raised to a parameter p>0 between x and the nearest mean m in M.
        3. Choose the next mean mx randomly from the set X, where the probability of a point x in X being chosen
            is proportional to D(x), and add mx to M.
        4. Repeat steps 2 and 3 a total of k − 1 times to produce k initial means.
        5. Apply the standard k-means algorithm, initialized with these means.

    Parameters available in all map and reduce functions
    ----------
    centroids : string
                Path to the file where the means in M are stored.

    p : string (representing a float)
        Parameter that determines how much the centroids are spread through the space.

    """

    def configure_args(self):
        super(MRJob_plusplus, self).configure_args()
        self.pass_arg_through('--runner')
        self.add_file_arg('--centroids')
        self.add_passthru_arg('--p', type=float, help='++ parameter')


    def map_distance(self, _, line):
        """Compute the distance raised to p between a point and the nearest mean m in M.

        Parameters
        ----------
        line: string
              Line of the .csv file given as input to MrJob. It corresponds to one point of the dataset.
              The first column is assumed to be the row index and last column is assumed to be the (true)
              classification label and they are therefore discarded.

        Returns
        -------
        _, (point, distance) : _, (ndarray, float)
                   The key is not specified in order to aggregate all couples in the same reducer.
                   The first element of the value is a list, representing the point, that may be converted to ndarray of shape (1, n_columns);
                   The second element is the distance raised to p between the point and the nearest mean m in M.

        """

        l = line.split(",")
        if len(l) == 1:
            return

        point = np.array([float(x) for x in l[1:-1]])

        # Read centroids from file
        with open(self.options.centroids, 'rb') as f:
            centroids = pickle.load(f)

        # Distance between points and nearest mean m
        distances = np.min([np.linalg.norm(point - c) for c in centroids])**self.options.p

        yield None, (point.tolist(), distances.tolist())


    def reduce_distance(self, _, pair):
        """Choose the next mean mx randomly from the set X, where the probability
        of a point x in X being chosen is proportional to D(x), and add mx to M.

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
        return [MRStep(mapper=self.map_distance, reducer=self.reduce_distance)]

if __name__ == '__main__':
    MRJob_plusplus.run()
