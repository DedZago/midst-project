import os
import sys
from time import time
from datetime import datetime

from src.file_io import *

def main():
    # create log.txt if it does not exist
    if not os.path.exists("log.txt"):
        col_names =  ["id", "file", "date", "algorithm", "initialization", "centroids (k)", "fuzzy weight (m)", "stopping_criterion", "init_time", "algorithm_time", "iteration"]
        with open("log.txt", "a") as log_file:
            log_file.write(",".join(col_names)+"\n")

    DIR = os.getcwd()
    data_file = DIR + "/" + sys.argv[1]

    # load settings from json file
    settings = load_settings("settings.json")
    if settings == False:
        return

    # repeat for each number of clusters
    for k in settings["centroids"]:

        # repeat for each initialization
        for initialization, initialization_name in settings["initialization"]:
            try:
                print("\n["+initialization_name+"]")
                init_start_time = time()
                centroids = initialization(fdata=data_file, k=k, p=settings["plusplus_parameter"], runner=settings["parallel"])
                init_time = time() - init_start_time
            except:
                # if initialization fails, go to the next one
                print("[ERROR]", initialization_name, "failed.")
                raise
                continue

            # # repeat for each stopping criterion
            for stopping_criterion, stopping_criterion_name in settings["stopping_criterion"]:

                # repeat for each algorithm
                for algorithm, algorithm_name in settings["algorithm"]:

                    # repeat for each fuzzy weighting exponent if the algorithm is fuzzy c-means
                    fuzzy_weighting_exponent = settings["fuzzy_weighting_exponent"] if algorithm_name=="fuzzy-cmeans" else ["-"]
                    for m in fuzzy_weighting_exponent:

                        # save initializated centroids in file
                        try:
                            alg_name = algorithm_name + str(m) if m != "-" else algorithm_name
                            centroids_file = "_".join([settings["id"], "centroids", alg_name, initialization_name, stopping_criterion_name, str(k)])
                            print("\n\n[" + centroids_file + "]")
                            centroids_file = DIR + "/centroids/" + centroids_file
                            centroids_to_disk(centroids, centroids_file)
                        except:
                            print("[ERROR]", centroids_file, " not saved succesfully.")
                            continue

                        algorithm_start_time = time()
                        # limit of settings["max_iterations"] iterations
                        i = 1
                        old_centroids = centroids
                        error_before_convergence = False
                        while i <= settings["max_iterations"]:
                            try:
                                print("Iteration", i, ";", end=" ")
                                new_centroids = algorithm(fdata=data_file, fcentroids=centroids_file, m=m, runner=settings["parallel"])
                                centroids_to_disk(new_centroids, centroids_file)

                                # check if stopping criterior is satisfied
                                diff = stopping_criterion(old_centroids, new_centroids)
                                print("stopping criterion:", diff) # "(stop if diff<="+str(limit)+")"
                                if diff > settings["stopping_criterion_threshold"]:
                                    old_centroids = new_centroids
                                    i += 1
                                else:
                                    print("Convergence reached in", i, "iterations.\n")
                                    break
                            except:
                                print("[ERROR] Iteration", i, "failed: the algorithm is interrupted.")
                                error_before_convergence = True
                                break
                        if not error_before_convergence:
                            algorithm_time = time() - algorithm_start_time
                            # if convergence is not reached in settings["max_iterations"] iterations, print this  message
                            if i > settings["max_iterations"]:
                                print("Convergence NOT reached in", settings["max_iterations"], "iterations.")

                            if settings["logging"] == True:
                                # information about the completed combination is appended in log.txt
                                n_iter = str(i-1)+"*" if i > settings["max_iterations"] else str(i)
                                info_list = [settings["id"], sys.argv[1], datetime.now().strftime("%Y/%m/%d %H:%M:%S"), algorithm_name, initialization_name, str(k), str(m), stopping_criterion_name, str(init_time), str(algorithm_time), n_iter]
                                with open("log.txt", "a") as log_file:
                                    log_file.write(",".join(info_list)+"\n")


if __name__ == '__main__':
    main()
