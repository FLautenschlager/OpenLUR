"""
Land-use regression with Neural Network from SKLearn.
"""

import sys
from os.path import join, basename, abspath, dirname
# This is necessary so that it is possible to import files from the parent dir
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import argparse
import ast
import re

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import numpy as np

from utils import paths
from hf_utils import load_input_file, write_results_file

# Default values for program arguments
INPUT_FILE_PATH = join(
    paths.extdatadir, 'pm_01072012_31092012_filtered_ha_200.csv')
FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']
RESULTS_FILE_NAME = 'skl_nn_output.csv'

HIDDEN_SIZE = 4
L2_SCALE = 0.000001
LEARNING_RATE = 0.1
STEPS = 4000

def cross_validation(X_t, y_t):
    kf = KFold(n_splits=10, shuffle=True)
    rsq = []
    rsq_train = []
    rmse = []
    rmse_train = []
    mae = []
    mae_train = []

    for train, test in kf.split(X_t):
        X_train, y_train = X_t[train, :], y_t[train]
        X_test, y_test = X_t[test, :], y_t[test]

        sc = StandardScaler()
        r = MLPRegressor(hidden_layer_sizes=(HIDDEN_SIZE, ), activation='relu', solver='adam', alpha=L2_SCALE, batch_size='auto', learning_rate='constant', learning_rate_init=LEARNING_RATE, power_t=0.5, max_iter=STEPS, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        pipe = Pipeline([('Scaler', sc),
                         ('Regressor', r)])
                         
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        pred_train = pipe.predict(X_train)

        rsq_train.append(r2_score(y_train, pred_train))
        rsq.append(r2_score(y_test, pred))
        mae_train.append(mean_absolute_error(y_train, pred_train))
        mae.append(mean_absolute_error(y_test, pred))
        rmse_train.append(np.sqrt(mean_squared_error(y_train, pred_train)))
        rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

    # print("R2-score on training folds = " + score_train)
    # print("R2-score on test folds = " + score)
    print("Mean R2-score on training data = {}".format(np.mean(rsq_train)))
    print("Mean R2-score on test data = {}".format(np.mean(rsq)))
    return {
        'rmse-train': np.mean(rmse_train),
        'rmse': np.mean(rmse),
        'mae-train': np.mean(mae_train),
        'mae': np.mean(mae),
        'rsq-train': np.mean(rsq_train),
        'rsq': np.mean(rsq)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path', help='File with calibration data',
                        nargs='?', default=INPUT_FILE_PATH)
    parser.add_argument('results_file', default=RESULTS_FILE_NAME, nargs='?',
                        help='File where to output the results')
    parser.add_argument('-f', '--feature_cols', default=FEATURE_COLS,
                        help='Feature columns to use for input')
    args = parser.parse_args()

    # Convert feature cols string to list
    if not isinstance(args.feature_cols, list):
        args.feature_cols = ast.literal_eval(args.feature_cols)

    if not isinstance(args.feature_cols, list):
        print('feature_cols is not a valid list')

    data = load_input_file(args.input_file_path)

    X_train = data[args.feature_cols].values
    y_train = data['pm_measurement'].values

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)

    # Parse timeframe from file name
    tf_pattern = re.compile('\d{8}_\d{8}')
    timeframe = tf_pattern.search(basename(args.input_file_path)).group(0)

    run_info = {
        'source': 'skl_nn',
        'feature_cols': args.feature_cols,
        'hidden_size': HIDDEN_SIZE,
        'learning_rate': LEARNING_RATE,
        'l2_scale': L2_SCALE,
        'steps': STEPS,
        'tiles': len(data),
        'timeframe': timeframe
    }

    print('Next Run:', run_info)

    # Do 10-fold cross validation on new data set
    results = cross_validation(X_train, y_train)

    # Merge run information with results
    results = {**run_info, **results}

    # Write results to file
    write_results_file(args.results_file, results)
    