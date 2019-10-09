import logging
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
#from multiprocessing import Pool
import multiprocessing as mp

from utils.MyPool import MyPool as Pool

from regression_runner import run_regression
from utils.DataLoader import Dataset
from utils.color import Color

logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s', level=logging.INFO)

features = [
    "os",
     "laei_small",
    # "laei",
     "both"
]
# feature_type=features[3]

modelnames = [
    #"AutoML",
    #"Random_Forest_random_search"#,
    #"Random_Forest_Standard",
     "GAM"
]
iterations = 10


def split_os(x, y):
    return train_test_split(x, y, test_size=0.1)

def split_laei(x, y, x_test, y_test, trainsize=180):
    # select 180 points from laei:
    idx = np.random.choice(x.shape[0], trainsize, replace=False)
    x_train_laei_split = x.loc[idx, :]
    y_train_laei_split = y[idx]

    # select 20 points from laei (test):
    idx = np.random.choice(x_test.shape[0], 20, replace=False)
    x_test_laei_split = x_test.loc[idx, :]
    y_test_laei_split = y_test[idx]
    return x_train_laei_split, x_test_laei_split, y_train_laei_split, y_test_laei_split

def run_regression_wrapper(input):
    model, x_train, y_train, x_test, y_test, iteration, writer = input
    return run_regression(model, x_train, y_train, x_test, y_test, iteration=iteration, tensorboard=writer)

def run(model, iterations=2, filename=None, season=1):
    x_train_laei, y_train_laei, x_test_laei, y_test_laei = Dataset.laeiOSM()
    x_train_os, y_train_os, x_test_os, y_test_os = Dataset.OpenSenseOSM(season)

    # Scaling
    laei_scaler = StandardScaler().fit(y_train_laei.reshape(-1, 1))
    os_scaler = StandardScaler().fit(y_train_os.reshape(-1, 1))
    y_train_laei = laei_scaler.transform(y_train_laei.reshape(-1, 1)).ravel()
    y_test_laei = laei_scaler.transform(y_test_laei.reshape(-1, 1)).ravel()
    y_train_os = os_scaler.transform(y_train_os.reshape(-1, 1)).ravel()
    #y_test_os = laei_scaler.transform(y_test_os.reshape(1, -1)).squeeze()

    logging.info("Start model {} on {}".format(model, feature_type))
    writer = SummaryWriter(comment="_{}_{}_{}iterations".format(feature_type, model, iterations))
    starttime = time.time()
    input = []
    results = []
    for i in range(iterations):

        # sample laei in 180/20
        x_train_laei_split, x_test_laei_split, y_train_laei_split, y_test_laei_split = split_laei(x_train_laei, y_train_laei, x_test_laei, y_test_laei)

        # split os in 180/20
        x_train_os_split, x_test_os_split, y_train_os_split, y_test_os_split = split_os(x_train_os, y_train_os)

        if feature_type == "both":
            # concatenate os and laei data
            x_train = np.concatenate((x_train_os_split, x_train_laei_split), axis=0)
            x_test = np.concatenate((x_test_os_split, x_test_laei_split), axis=0)
            y_train = np.concatenate((y_train_os_split, y_train_laei_split), axis=0)
            y_test = np.concatenate((y_test_os_split, y_test_laei_split), axis=0)
        elif feature_type == "laei_small":
            x_train = x_train_laei_split
            y_train = y_train_laei_split
            x_test = x_test_laei_split
            y_test = y_test_laei_split
        elif feature_type == "laei":
            x_train = x_train_laei
            y_train = y_train_laei
            x_test = x_test_laei
            y_test = y_test_laei
        elif feature_type == "os":
            x_train = x_train_os_split
            y_train = y_train_os_split
            x_test = x_test_os_split
            y_test = y_test_os_split
        else:
            break

        input.append((model, x_train, y_train, x_test, y_test, i+1, writer))

        results.append(run_regression_wrapper(input[-1]))
        print(results[-1])

    results = pd.concat(results, ignore_index=True)
    timediff = time.time() - starttime

    logging.info(
        Color.BOLD + Color.GREEN + "Results  for model {}, time needed: {:.2f} minutes, mean R2 on test: {:.2f}:".format(
            model, timediff / 60, results[results["type"] == "test"]["r2"].mean()) + Color.END)
    logging.debug(results.groupby("type").r2.mean())

    if filename:
        pickle.dump(results, open(filename, "wb"))
        logging.info("saved in {}".format(filename))
    logging.info(" ")
    logging.info(" ")


if __name__ == "__main__":
    logging.info("start")

    for feature_type in features:
        for model in modelnames:
            # run_londondata(model, iterations=iterations, filename="output/{}_longrun.p".format(model))
            run(model, iterations=iterations,
                        filename="output/{}_train_{}_{}_iterations.p".format(model, feature_type, iterations))
