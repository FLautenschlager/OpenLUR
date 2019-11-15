import logging
import time
import argparse
from tqdm import tqdm
from scipy.stats import wilcoxon
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

from pygam import LinearGAM
from utils.DataLoader import Dataset

logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s', level=logging.INFO)

features = [
    #"os",
    # "laei_small",
     "laei",
    # "both"
]

def check_significance(x, y, x_test, cols, col_add):
    if len(cols)==0:
        return True
    model = LinearGAM().gridsearch(x[cols].values, y, progress=False)
    predictions1 = model.predict(x_test[cols].values)
    model = LinearGAM().gridsearch(x[cols + [col_add]].values, y, progress=False)
    predictions2 = model.predict(x_test[cols + [col_add]].values)
    test_stats = wilcoxon(predictions1, predictions2)

    return test_stats.pvalue < 0.05

def feature_selection_single(x, y, x_test, y_test):
    timestart = time.time()
    cols = list(deepcopy(x.columns))
    best_result = 0
    selected_cols = []
    continue_selection = True

    iterationresult = {}
    while continue_selection:
        for col in tqdm(cols, leave=False):
            testcols = selected_cols + [col]

            model = LinearGAM().gridsearch(x[testcols].values, y, progress=False)
            iterationresult[col] = model._estimate_r2(x_test[testcols].values, y_test)['explained_deviance']
            #iterationresult[col] = r2_score(model.predict(x_test[testcols].values), y_test)

        key = max(iterationresult.keys(), key=(lambda key: iterationresult[key]))
        if (iterationresult[key] > best_result) & check_significance(x, y, x_test, selected_cols, key):
            best_result = iterationresult[key]
            selected_cols.append(key)
            cols.remove(key)
        else:
            continue_selection = False

    logging.info("{}: {}".format(selected_cols, best_result))
    return best_result, selected_cols, time.time() - timestart


def split_laei(x, y, number_train=180, number_test=20):

    if number_train == -1:
        complete = x.shape[0]
        split = 0.2
    else:
        complete = number_test+number_train
        split = number_test/complete

    idx = np.random.choice(x.shape[0], complete, replace=False)
    x_laei = x.loc[idx, :]
    y_laei = y[idx]
    return train_test_split(x_laei, y_laei, test_size=split)

def split_os(x, y):
    return train_test_split(x, y, test_size=0.1)


def load_data(dataset, numlaei=180):
    x_train_laei, y_train_laei, x_test_laei, y_test_laei = Dataset.laeiOSM()
    x_train_os, y_train_os, x_test_os, y_test_os = Dataset.OpenSenseOSM(1)

    # Scaling
    laei_scaler = StandardScaler().fit(y_train_laei.reshape(-1, 1))
    os_scaler = StandardScaler().fit(y_train_os.reshape(-1, 1))
    y_train_laei = laei_scaler.transform(y_train_laei.reshape(-1, 1)).ravel()
    y_test_laei = laei_scaler.transform(y_test_laei.reshape(-1, 1)).ravel()
    y_train_os = os_scaler.transform(y_train_os.reshape(-1, 1)).ravel()
    #y_test_os = laei_scaler.transform(y_test_os.reshape(1, -1)).squeeze()

    # sample laei in 180/20
    x_train_laei_split, x_test_laei_split, y_train_laei_split, y_test_laei_split = split_laei(x_train_laei,
                                                                                              y_train_laei,
                                                                                              number_train=numlaei)

    # split os in 180/20
    x_train_os_split, x_test_os_split, y_train_os_split, y_test_os_split = split_laei(x_train_os, y_train_os, number_train=180, number_test=20)

    if dataset == "both":
        # concatenate os and laei data
        x_train = pd.concat((x_train_os_split, x_train_laei_split), axis=0, ignore_index=True)
        x_test = pd.concat((x_test_os_split, x_test_laei_split), axis=0, ignore_index=True)
        y_train = np.concatenate((y_train_os_split, y_train_laei_split), axis=0)
        y_test = np.concatenate((y_test_os_split, y_test_laei_split), axis=0)
    elif dataset == "laei_small":
        x_train = x_train_laei_split
        y_train = y_train_laei_split
        x_test = x_test_laei_split
        y_test = y_test_laei_split
    elif dataset == "laei":
        x_train = x_train_laei_split
        y_train = y_train_laei_split
        x_test = x_test_laei_split
        y_test = y_test_laei_split
    elif dataset == "os":
        x_train = x_train_os_split
        y_train = y_train_os_split
        x_test = x_test_os_split
        y_test = y_test_os_split
    else:
        logging.error("Unknown dataset: {}".format(dataset))
        x_train = 0
        y_train = 0
        x_test = 0
        y_test = 0

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    logging.info("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset: one of os, laei, laei_small, both", type=str)
    parser.add_argument("--iterations", "-i", help="how many iterations to make", type=int, default=40)
    parser.add_argument("--number_of_laei", "-n", help="for both and laei datasets: how many datapoints to sample from laei", type=int, default=180)

    args = parser.parse_args()


    results = []
    for i in range(args.iterations):
        x, y, x_test, y_test = load_data(args.dataset, numlaei=args.number_of_laei)
        logging.info("Starting iteration {} with {}/{} samples".format(i+1, x.shape[0], x_test.shape[0]))
        results.append(feature_selection_single(x, y, x_test, y_test))

    df = pd.DataFrame(results, columns=["r2", "features", "time needed"])
    filename = "runs/featureselection/{}_{}iterations_r2sklearn.csv".format(args.dataset, args.iterations)
    if os.path.isfile(filename):
        logging.info("file \"{}\" already exists.".format(filename))
        filenamestump = filename[:-4] + "_{}.csv"
        i=1
        filename = filenamestump.format(i)
        while os.path.isfile(filename):
            logging.info("file \"{}\" already exists.".format(filename))
            i += 1
            filename = filenamestump.format(i)
    logging.info("Save as \"{}\".".format(filename))
    df.to_csv(filename, index=False)