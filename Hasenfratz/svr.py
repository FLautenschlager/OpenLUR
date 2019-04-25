"""
Land-use regression with support vector regression.
"""

import sys
from os.path import join, basename, abspath, dirname
# This is necessary so that it is possible to import files from the parent dir
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import argparse
import ast
import re

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import numpy as np

from experiments import paths
from hf_utils import load_input_file, write_results_file, is_in, interpolate

# Default values for program arguments
INPUT_FILE_PATH = join(
    paths.extdatadir, 'pm_01072012_31092012_filtered_ha_200.csv')
FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']
RESULTS_FILE_NAME = 'svr_output.csv'

KERNEL = 'rbf'
GAMMA = 'auto'
C = 1.0
EPSILON = 0.1
DEGREE = 3


def cross_validation(data, kernel, gamma, c, epsilon, degree,
                     interpolation_factor):
    kf = KFold(n_splits=10, shuffle=True)
    rsq = []
    rsq_train = []
    rmse = []
    rmse_train = []
    mae = []
    mae_train = []

    # Get the original unshifted grid and split the CV based on this
    og_grid_data = data[(data['y'] % 100 == 0) & (data['x'] % 100 == 0)]

    for _, test in kf.split(og_grid_data):
        test_data = data.iloc[test]

        # Gather all cells that do not overlap with a test cell for training
        train_data = data[data.apply(
            lambda c: not is_in(c, test_data), axis=1)]

        # Interpolate new rows for train_calib_data
        train_data = interpolate(train_data, int(
            interpolation_factor * len(train_data)))

        X_train = train_data[args.feature_cols].values
        y_train = train_data['pm_measurement'].values
        X_test = test_data[args.feature_cols].values
        y_test = test_data['pm_measurement'].values

        X_train = np.ascontiguousarray(X_train)
        y_train = np.ascontiguousarray(y_train)
        X_test = np.ascontiguousarray(X_test)
        y_test = np.ascontiguousarray(y_test)

        sc = StandardScaler()

        r = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=0.0, tol=0.001,
                C=c, epsilon=epsilon, shrinking=True, cache_size=200,
                verbose=False, max_iter=-1)

        pipe = Pipeline([('Scaler', sc), ('Regressor', r)])

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
    parser.add_argument('-i', '--interpolation_factor', type=float, default=0.0,
                        help='Number of rows that should be generated through' +
                        ' interpolation as a percentage of train data length ' +
                        '(example: len(train_data) = 200 and -i = 1 -> 200 ' +
                        'interpolated rows, 400 rows overall)')
    # Hyperparameters
    parser.add_argument('-k', '--kernel', default=KERNEL,
                        help='Kernel')
    parser.add_argument('-g', '--gamma', default=GAMMA,
                        help='Kernel coefficient')
    parser.add_argument('-c', default=C, type=float,
                        help='Penalty parameter C of the error term')
    parser.add_argument('-e', '--epsilon', default=EPSILON, type=float,
                        help='Specifies the epsilon-tube')
    parser.add_argument('-d', '--degree', default=DEGREE, type=int,
                        help='Degree of polynomial kernel function')

    parser.add_argument('-en', '--experiment_number', default=0, type=int,
                        help='Experiment number')
    args = parser.parse_args()

    # Convert feature cols string to list
    if not isinstance(args.feature_cols, list):
        args.feature_cols = ast.literal_eval(args.feature_cols)

    if not isinstance(args.feature_cols, list):
        print('feature_cols is not a valid list')

    try:
        args.gamma = float(args.gamma)
    except Exception:
        pass

    data = load_input_file(args.input_file_path)

    # Parse timeframe from file name
    tf_pattern = re.compile('\d{8}_\d{8}')
    timeframe = tf_pattern.search(basename(args.input_file_path)).group(0)

    run_info = {
        'source': 'svr',
        'kernel': args.kernel,
        'gamma': args.gamma,
        'c': args.c,
        'epsilon': args.epsilon,
        'degree': args.degree,
        'feature_cols': args.feature_cols,
        'tiles': len(data),
        'timeframe': timeframe,
        # For hyperparameter search
        'experiment_number': args.experiment_number
    }

    print('Next Run:', run_info)

    # Do 10-fold cross validation on new data set
    results = cross_validation(data, args.kernel, args.gamma, args.c,
                               args.epsilon, args.degree,
                               args.interpolation_factor)

    # Merge run information with results
    results = {**run_info, **results}

    # Write results to file
    write_results_file(args.results_file, results)
