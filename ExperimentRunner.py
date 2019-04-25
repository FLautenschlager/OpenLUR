import pickle
import time

import pandas as pd

from regression_runner import run_regression, models
from utils.DataLoader import Dataset
from utils.MyPool import MyPool
from utils.color import Color


def run_londondata_mapper(args):
    model, iterations = args
    run_londondata(model, iterations=iterations)

def run_londondata(model, iterations=2, filename=None):

    x_train, y_train, x_test, y_test = Dataset.laeiOSM()

    print("Start model {}".format(model))

    starttime = time.time()
    results = []
    for i in range(iterations):
        results.append(run_regression(model, x_train, y_train, x_test, y_test))

    results = pd.concat(results, ignore_index=True)
    timediff = time.time()-starttime

    search_results = results[results.type == "search"]
    print(Color.BOLD + Color.GREEN + "Meaned over {} iterations ({} minutes each), the model {} reached a RMSE of {}, MAE of {} and R2 of {}.".format(iterations, timediff/60/iterations, model, search_results.rmse.mean(), search_results.mae.mean(), search_results.r2.mean()) + Color.END)
    if filename:
        pickle.dump(results, open(filename, "wb"))


if __name__=="__main__":
    model = "Random_Forest_random_search"
    #model = "AutoML"

    modelnames = ["Random_Forest_random_search", "AutoML", "Random_Forest_Standard"]
    iterations = 2
    #run_londondata(model, iterations=5)
    for model in modelnames:
    #    newpid = os.fork()
    #    if newpid == 0:
    #        print("Start next model: {}".format(model))
    #    else:
         run_londondata(model, iterations=iterations)
    #        break

    #if newpid==0:
    #    print("Started all models.")

    #inputs = []
    #for m in modelnames:
    #    inputs.append((m, iterations))
    #pool = MyPool(processes=len(modelnames))
    #pool.map(run_londondata_mapper, inputs)
