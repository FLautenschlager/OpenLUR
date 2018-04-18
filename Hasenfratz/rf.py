"""
Land-use regression with RandomForest.
"""

import sys
from os.path import join, basename, abspath, dirname
# This is necessary so that it is possible to import files from the parent dir
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import argparse
import ast
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import numpy as np

from utils import paths
from hf_utils import load_input_file, write_results_file

# Default values for program arguments
INPUT_FILE_PATH = join(
    paths.extdatadir, 'pm_01042012_30062012_filtered_ha_200.csv')
FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']
RESULTS_FILE_NAME = 'random_forest_output.csv'

N_ESTIMATORS = 100
MAX_FEATURES = 'auto'
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
BOOTSTRAP = True


def cross_validation(X_t, y_t, n_estimators, max_features, max_depth,
                     min_samples_split, min_samples_leaf, bootstrap):
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

        r = RandomForestRegressor(n_estimators=n_estimators, criterion='mse',
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=max_features,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None, bootstrap=bootstrap,
                                  oob_score=False, n_jobs=1, random_state=None,
                                  verbose=0, warm_start=False)

        pipe = Pipeline([('Imputer', im), ('Scaler', mm),
                         #  ('Polynomial', p),
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
    # Hyperparameters
    parser.add_argument('-n', '--n_estimators', default=N_ESTIMATORS, type=int,
                        help='Number of trees in the forest')
    parser.add_argument('-mf', '--max_features', default=MAX_FEATURES,
                        help='Max number of features considered for splitting' +
                        ' a node')
    parser.add_argument('-md', '--max_depth', default=MAX_DEPTH,
                        help='Max number of levels in each decision tree')
    parser.add_argument('-mss', '--min_samples_split', default=MIN_SAMPLES_SPLIT,
                        type=int, help='Min number of data points placed in a' +
                        ' node before the node is split')
    parser.add_argument('-msl', '--min_samples_leaf', default=MIN_SAMPLES_LEAF,
                        type=int, help='Min number of data points allowed in ' +
                        'a leaf node')
    parser.add_argument('-b', '--bootstrap', default=BOOTSTRAP,
                        help='Sample data points with or without replacement')

    parser.add_argument('-en', '--experiment_number', default=0, type=int,
                        help='Experiment number')
    args = parser.parse_args()

    # Convert feature cols string to list
    if not isinstance(args.feature_cols, list):
        args.feature_cols = ast.literal_eval(args.feature_cols)

    if not isinstance(args.feature_cols, list):
        print('feature_cols is not a valid list')

    try:
        args.max_features = int(args.max_features)
    except Exception:
        pass

    try:
        args.max_depth = int(args.max_depth)
    except Exception:
        pass

    args.bootstrap = args.bootstrap == 'true' or args.bootstrap == 'True' or args.bootstrap == '1'

    data = load_input_file(args.input_file_path)

    X_train = data[args.feature_cols].values
    y_train = data['pm_measurement'].values

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)

    # Parse timeframe from file name
    tf_pattern = re.compile('\d{8}_\d{8}')
    timeframe = tf_pattern.search(basename(args.input_file_path)).group(0)

    run_info = {
        'source': 'random_forest',
        'feature_cols': args.feature_cols,
        'tiles': len(data),
        'timeframe': timeframe,
        'n_estimators': args.n_estimators,
        'max_features': args.max_features,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'bootstrap': args.bootstrap,
        # For hyperparameter search
        'experiment_number': args.experiment_number
    }

    print('Next Run:', run_info)

    # Do 10-fold cross validation on new data set
    results = cross_validation(X_train, y_train, args.n_estimators,
                               args.max_features, args.max_depth,
                               args.min_samples_split, args.min_samples_leaf,
                               args.bootstrap)

    # Merge run information with results
    results = {**run_info, **results}

    # Write results to file
    write_results_file(args.results_file, results)
