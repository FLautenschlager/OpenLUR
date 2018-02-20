"""
Land-use regression with Lasso.
"""

import sys
from os.path import join, basename, abspath, dirname
# This is necessary so that it is possible to import files from the parent dir
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import argparse
import ast
import re

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import numpy as np

from utils import load_input_file, write_results_file, paths

# Default values for program arguments
INPUT_FILE_PATH = join(
    paths.extdatadir, 'pm_01072012_31092012_filtered_ha_200.csv')
FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']
RESULTS_FILE_NAME = 'lasso_output.csv'


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

        im = Imputer(strategy='most_frequent')
        mm = MinMaxScaler()
        p = PolynomialFeatures(
            degree=3, interaction_only=False, include_bias=True)

        r = Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                  precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
                  warm_start=False, positive=False, random_state=None,
                  selection='cyclic')

        pipe = Pipeline([('Imputer', im), ('Scaler', mm),
                         ('Polynomial', p), ('Regressor', r)])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        pred_train = pipe.predict(X_train)

        rsq_train.append(r2_score(y_train, pred_train))
        rsq.append(r2_score(y_test, pred))
        mae_train.append(mean_absolute_error(y_train, pred_train))
        mae.append(mean_absolute_error(y_test, pred))
        rmse_train.append(mean_squared_error(y_train, pred_train))
        rmse.append(mean_squared_error(y_test, pred))

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
        'source': 'lasso',
        'feature_cols': args.feature_cols,
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
