import csv
import os
import numpy as np
import pickle
import re
import sys

from src.mrjob.membership_matrix import MRJob_membership_matrix

'''
python centroids_to_mb.py <data> <centroids> <fuzzy_weight_exponent>
'''

def main(fdata, fcentroids):
    """Wrapper to call a MapReduce job via MRJob that create the membership matrix.
    The update formula of the membership matrix of fuzzy c-means algorithm is used.


    Parameters
    ----------
    fdata : string
            Path to the .csv file containing n_rows rows of observations and n_columns columns of features.
            The first column is assumed to be the row index and last column is assumed to be the (true)
            classification label and they are therefore discarded.

    fcentroids : string
                 Name of the file where centroids are stored; it must be in centroids folder.

    Returns
    -------
    None

    Notes
    -----
    The weight exponent, necessary for the update formula, is extracted from fcentroids, therefore the
    name of the file must be the one assigned by main.py. The file containing the membership matrix is
    stored in membership folder and its name is built by replacing 'centroids' by membership in fcentroids.

    """

    # Get the name of the file that will contain the membership matrix
    mb = fcentroids.split("_")
    mb[-5] = "membership"
    membership_matrix_file = "_".join(mb)

    # Get weighting exponent (m)
    m = mb[-4].replace("fuzzy-cmeans", "")

    # Get complete path of each file
    DIR = os.getcwd()
    fdata = DIR + "/" + fdata
    fcentroids = DIR + "/centroids/" + fcentroids
    membership_matrix_file = DIR + "/membership/" + membership_matrix_file

    # Check if data file exist
    if not os.path.exists(fdata):
        print("[ERROR]", fdata, "does not exist.")
        return False

    # Check if centroids file exist
    if not os.path.exists(fcentroids):
        print("[ERROR]", fcentroids, "does not exist.\nNote that the file is searched in centroids folder.")
        return False

    # Check if the name of centroids file is compatible
    pattern = re.compile(".*_centroids_fuzzy-cmeans([0-9]+|[0-9]+.[0-9]+)_[^_]+_[^_]+_[0-9]+")
    if not pattern.match(fcentroids) or len(centroids_file.split("_centroids_")[1].split("_")) != 4:
        print("[ERROR]", fcentroids, "is a name not compatible. Compatible name:\n<id>_centroids_<algorithm>_<initialization>_<stop_crit>_<k> without '_'s in the parameters (<.>)")
        return False

    try:
        membership_matrix_job = MRJob_membership_matrix(args=[fdata, "--runner", "local", "--centroids", fcentroids, "--weight", m, "--no-bootstrap-mrjob"])
        with membership_matrix_job.make_runner() as membership_matrix_runner:
            membership_matrix_runner.run()
            # store membership matrix in file
            for key, value in membership_matrix_job.parse_output(membership_matrix_runner.cat_output()):
                with open(membership_matrix_file, "ab") as f:
                    pickle.dump(value, f)
        return True
    except:
        print("[ERROR] Something goes wrong during MrJob.")
        raise
        return False


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Wrong inputs, type something similar to:\npython centroids_to_mb.py data/DATA.CSV centroids/CENTROIDS WEIGHT NEW_FILE_NAME")
    else:
        fdata = sys.argv[1]
        fcentroids = sys.argv[2]

        if main(fdata, fcentroids):
            print("\nFile created correctly.")
