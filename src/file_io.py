import json
import numpy as np
import os
import pickle
import re

def extract_mrjob(job, runner, which="value"):
    """Extract objects yielded by a MRJob runner.

    Parameters
    ----------
    job : MRJob or children class
          MRJob object which yields a (key, value) pair.

    runner : MRJob runner
             Object created by MRJob.make_runner() which runs the job.

    which : string
            If "value" returns value, if "key" returns key, if "both" returns both.

    Returns
    -------
    c : object or (object, object), depending on which is selected.
    
    """

    c = []
    for key,value in job.parse_output(runner.cat_output()):
        #print(key,value)
        if which == "key":
            c.append(key)
        elif which == "value":
            c.append(value)
        elif which == "both":
            c.append((key, value))
    return c


def centroids_to_disk(centroids, fname):
    """Saves centroids to a file.

    Parameters
    ----------
    centroids : ndarray
                Matrix of centroids to save to disk.

    fname : string
            Name of file where centroids must be saved.

    Returns
    -------
    None

    """

    print("Saving file of shape:", np.array(centroids).shape)
    #print(np.array(centroids))
    f = open(fname, "wb")
    pickle.dump(centroids, f)
    f.close()


def load_settings(fname):
    """Load algorithm settings from json file. Return False if there are wrong settings
       or the JSON file is not found.

    Parameters
    ----------
    fname : string
            String with the name of the json file where settings are located.

    Returns
    -------
    settings : dict
               Keys are setting names, values are appropriately constructed for the algorithms,
               depending on input. If an error occurred, False is returned.

    """

    if not os.path.exists("settings.json"):
        print("settings.json not found.")
        return False
    else:
        print("Loading settings...")
        with open(fname) as json_file:
            settings = json.load(json_file)
            # ---
            if "fuzzy-cmeans" not in settings["algorithm"]:
                # not used if "fuzzy-cmeans" is not in settings["algorithm"] but necessary for the pipeline
                settings["fuzzy_weighting_exponent"] = ["-"]
            else:
                if "fuzzy_weighting_exponent" not in settings:
                    print("[fuzzy_weighting_exponent] Fuzzy weighting exponent is not specified.")
                    return False
                else:
                    for m in settings["fuzzy_weighting_exponent"]:
                        if not isinstance(m, (int, float)) or m < 1:
                            print("[fuzzy_weighting_exponent] Fuzzy weighting exponent is not a number equal or greater than 1.")
                            return False
                # remove duplicates and sort the list
                settings["fuzzy_weighting_exponent"] = sorted(list(set(settings["fuzzy_weighting_exponent"])))
            # ---
            if "centroids" not in settings:
                print("[centroids] Number of centroids is not specified.")
                return False
            else:
                k_list = []
                for val in settings["centroids"]:
                    if isinstance(val, int) or str(val).isdigit():
                        k_list.append(int(val))
                    elif isinstance(val, str) and re.compile("[0-9]+-[0-9]+").match(val):
                        lower, upper = int(val.split("-")[0]), int(val.split("-")[1]) + 1
                        k_list.extend(range(lower, upper))
                    else:
                        print("[centroids] An element is specified incorrectly.")
                        return False
                # remove duplicates and sort the list
                settings["centroids"] = sorted([k for k in list(set(k_list)) if k > 0])
                if len(settings["centroids"]) == 0:
                    print("[centroids] Empty list of number of centroids.")
                    return False
            # ---
            if "plusplus" in settings["initialization"]:
                if "plusplus_parameter" not in settings:
                    print("[plusplus_parameter] The parameter of ++ initialization is not specified.")
                    return False
                elif not isinstance(settings["plusplus_parameter"], (int,float)) or settings["plusplus_parameter"] < 0:
                    print("[plusplus_parameter] The parameter of ++ initialization is not a positive number.")
                    return False
            # ---
            if "initialization" not in settings:
                print("[initialization] Initialization is not specified.")
                return False
            elif not isinstance(settings["initialization"], list):
                print("[initialization] Initialization is not a list.")
                return False
            else:
                for i in range(len(settings["initialization"])):
                    if settings["initialization"][i] == "plusplus":
                        from src.initialization import init_plusplus
                        settings["initialization"][i] = (init_plusplus, "plusplus")
                    elif settings["initialization"][i] == "step":
                        from src.initialization import init_step
                        settings["initialization"][i] = (init_step, "step")
                    elif settings["initialization"][i] == "random":
                        from src.initialization import init_random
                        settings["initialization"][i] = (init_random, "random")
                    else:
                        print("Unknown initialization:", settings["initialization"][i])
                        return False
            # ---
            if "algorithm" not in settings:
                print("[algorithm] Algorithm is not specified.")
                return False
            elif not isinstance(settings["algorithm"], list):
                print("[algorithm] Algorithm is not a list.")
                return False
            else:
                for i in range(len(settings["algorithm"])):
                    if settings["algorithm"][i] == "fuzzy-cmeans":
                        from src.algorithm import fuzzy_cmeans
                        settings["algorithm"][i] = (fuzzy_cmeans, "fuzzy-cmeans")
                    elif settings["algorithm"][i] == "kmeans":
                        from src.algorithm import kmeans
                        settings["algorithm"][i] = (kmeans, "kmeans")
                    else:
                        print("Unknown algorithm:", settings["algorithm"][i])
                        return False
            # ---
            if "stopping_criterion" not in settings:
                print("[stopping_criterion] Stopping criterion is not specified.")
                return False
            elif not isinstance(settings["stopping_criterion"], list):
                print("[stopping_criterion] Stopping criterion is not a list.")
                return False
            else:
                for i in range(len(settings["stopping_criterion"])):
                    if settings["stopping_criterion"][i] == "biggest-diff":
                        from src.stopping_criterion import biggest_diff
                        settings["stopping_criterion"][i] = (biggest_diff, "biggest-diff")
                    else:
                        print("Unknown stopping criterion:", settings["stopping_criterion"][i])
                        return False
            # ---
            if "stopping_criterion_threshold" not in settings:
                print("[stopping_criterion_limit] The threshold of the stopping criterion is not specified.")
                return False
            elif not isinstance(settings["stopping_criterion_threshold"], (int,float)) or settings["stopping_criterion_threshold"] < 0:
                print("[stopping_criterion_limit] The threshold of the stopping criterion is not a positive number.")
                return False
            # ---
            if "max_iterations" not in settings:
                print("[max_iterations] Limit of iterations is not specified: 50 is used as default.")
                settings["max_iterations"] = 50
            elif not isinstance(settings["max_iterations"], int):
                print("[max_iterations] Limit of iterations is not an integer.")
                return False
            # ---
            if "parallel" not in settings or settings["parallel"] != True:
                print("[parallel] parallel is not specified or it is specified incorrectly: False (not parallel) is used as default.")
                settings["parallel"] = "inline"
            else:
                settings["parallel"] = "local"
            # ---
            if "logging" not in settings or settings["logging"] != True:
                print("[logging] logging is not specified or it is specified incorrectly: False (no logs) is used as default.")
                settings["logging"] = False

        print("Settings loaded")

        return settings
