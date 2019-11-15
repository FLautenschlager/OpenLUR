import logging
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter

from ExperimentRunner import split_laei, split_os, run_regression_wrapper
from utils.DataLoader import Dataset
from utils.color import Color

# from multiprocessing import Pool

logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s', level=logging.INFO)

features = "both"
#modelnames = "Random_Forest_random_search"
iterations = 40

modelnames = "AutoML"

def run(iterations=2, filename=None, season=1, laei_size=180):
    model = "Random_Forest_random_search"
    feature_type = "both"

    x_train_laei, y_train_laei, x_test_laei, y_test_laei = Dataset.laeiOSM()
    x_train_os, y_train_os, x_test_os, y_test_os = Dataset.OpenSenseOSM(season)

    # Scaling
    laei_scaler = StandardScaler().fit(y_train_laei.reshape(-1, 1))
    os_scaler = StandardScaler().fit(y_train_os.reshape(-1, 1))
    y_train_laei = laei_scaler.transform(y_train_laei.reshape(-1, 1)).ravel()
    y_test_laei = laei_scaler.transform(y_test_laei.reshape(-1, 1)).ravel()
    y_train_os = os_scaler.transform(y_train_os.reshape(-1, 1)).ravel()
    #y_test_os = laei_scaler.transform(y_test_os.reshape(1, -1)).squeeze()

    logging.info("Start model {} on {} with {} training examples from laei".format(model, feature_type, laei_size))
    writer = SummaryWriter(comment="_{}".format(filename.split("/")[-1][:-2]))
    starttime = time.time()
    input = []
    results = []
    for i in range(iterations):

        # sample laei in 180/20
        x_train_laei_split, x_test_laei_split, y_train_laei_split, y_test_laei_split = split_laei(x_train_laei, y_train_laei, x_test_laei, y_test_laei, trainsize=laei_size)

        # split os in 180/20
        x_train_os_split, x_test_os_split, y_train_os_split, y_test_os_split = split_os(x_train_os, y_train_os)

        x_train = np.concatenate((x_train_os_split, x_train_laei_split), axis=0)
        x_test = x_test_laei.values
        y_train = np.concatenate((y_train_os_split, y_train_laei_split), axis=0)
        y_test = y_test_laei

        input.append((model, x_train, y_train, x_test, y_test, i+1, writer))

        results.append(run_regression_wrapper(input[-1]))
        print(results[-1])

    results = pd.concat(results, ignore_index=True)
    timediff = time.time() - starttime

    logging.info(Color.BOLD + Color.GREEN + "Results  for model {}, time needed: {:.2f} minutes, mean R2 on test: {:.2f}:".format(model, timediff / 60, results[results["type"] == "test"]["r2"].mean()) + Color.END)
    logging.debug(results.groupby("type").r2.mean())

    if filename:
        pickle.dump(results, open(filename, "wb"))
        logging.info("saved in {}".format(filename))
    logging.info(" ")
    logging.info(" ")


if __name__ == "__main__":
    logging.info("start")
    for train_size in [20,40,60,80,100,150,200,300,400,500,1000]:
    #train_size = 20
        run(iterations=iterations, filename="output/two_cities_experiment_{}laei_{}iterations.p".format(train_size, iterations), laei_size=train_size)
