"""
Land-use regression with GAM. Does a single ten-fold cross-validation but the
folds are calculated in parallel.
"""

import sys
from os.path import join, basename, abspath, dirname
# This is necessary so that it is possible to import files from the parent dir
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import ast
import re
import argparse
from multiprocessing import Pool, cpu_count

import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

from utils import paths
from hf_utils import load_input_file, write_results_file, is_in, interpolate

# Default values for program arguments
INPUT_FILE_PATH = join(
    paths.extdatadir, 'pm_01072012_31092012_filtered_ha_200.csv')
MODEL_VAR_PATH = join(paths.rdir, 'model_ha_variables.mat')
FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']
RESULTS_FILE_NAME = 'gam_output.csv'


def build_formula_string(feature_cols):
    """ Build formula for GAM depending on which features are used """

    smooth_template = 's({feature_col},bs="cr",k=3)'
    formula = 'pm_measurement~'

    for feature_col in feature_cols:
        # streetsize is not smoothed in Hasenfratz's original work
        if feature_col is 'streetsize':
            formula += feature_col + '+'
        else:
            formula += smooth_template.format(feature_col=feature_col) + '+'

    # Remove the last '+'
    return formula[:-1]


def calculate_gam(inputs):
    """ Calculate the results of a GAM """

    train_calib_data, test_calib_data, test_model_var, feature_cols = inputs

    # mgcv is the R package with the GAM implementation
    mgcv = importr('mgcv')
    base = importr('base')
    stats = importr('stats')

    # Activate implicit conversion of pandas to rpy2 and vice versa
    pandas2ri.activate()

    # This is the formula for the GAM
    # From https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.terms.html:
    # "s()" defines a smooth term in R
    # "bs" is the basis of the used smooth class
    # "cr" declares a cubic spline basis
    # "k" defines the dimension of the basis (upper limit on degrees of freedom)
    # formula = robjects.Formula('pm_measurement~s(industry,bs="cr",k=3)' +
    #     '+s(floorlevel,bs="cr",k=3)+s(elevation,bs="cr",k=3)' +
    #     '+s(slope,bs="cr",k=3)+s(expo,bs="cr",k=3)+streetsize' +
    #     '+s(traffic_tot,bs="cr",k=3)+s(streetdist_l,bs="cr",k=3)')
    formula = robjects.Formula(build_formula_string(feature_cols))
    # Hasenfratz uses a Gamma distribution with a logarithmic link
    family = stats.Gamma(link='log')

    # Train model
    model = mgcv.gam(formula, family, data=train_calib_data)
    su = base.summary(model)

    # Predict the test data
    pred_data = stats.predict(model, newdata=test_model_var, type='response')
    test_model_var_predictions = test_model_var.assign(prediction=pred_data)

    # Create a DataFrame where it is easily possible to compare measurements and
    # predictions
    test_measure_predict = test_calib_data.merge(
        test_model_var_predictions[['y', 'x', 'prediction']], how='inner', on=['y', 'x']
    )  # [['x', 'y', 'pm_measurement', 'prediction']]
    # Check how large the error is with the remaining 10% of data
    error_model = test_measure_predict['pm_measurement'] - \
        test_measure_predict['prediction']
    # Drop all NaN's
    error_model = error_model.dropna()
    # Calculate Root-mean-square error model
    rmse = np.sqrt(np.mean(error_model**2))
    # Calculate mean-absolute error
    mae = np.mean(np.abs(error_model))
    # Get R² from summary
    rsq = su.rx2('r.sq')[0]
    devexpl = su.rx2('dev.expl')[0]
    # Calculate Factor of 2
    fac2_ind = test_measure_predict['pm_measurement'] / \
        test_measure_predict['prediction']
    fac2_ind = fac2_ind[(fac2_ind <= 2) & (fac2_ind >= 0.5)].dropna()
    fac2 = (len(fac2_ind) / len(test_measure_predict['pm_measurement']) * 100)

    # calculate R2 between predicted and measured concentration
    r2val_formula = robjects.Formula('measurements~predictions')
    r2val_env = r2val_formula.environment
    r2val_env['measurements'] = test_measure_predict['pm_measurement']
    r2val_env['predictions'] = test_measure_predict['prediction']
    lt1 = stats.lm(r2val_formula)
    rsqval = base.summary(lt1).rx2('r.squared')[0]

    # Calculate adjusted R²: rsq-(1-rsq)*p/(n-p-1)
    p = len(feature_cols)
    n = len(train_calib_data) + len(test_calib_data)
    adj_rsqval = rsqval - (1 - rsqval) * p / (n - p - 1)

    # Return metrics and predicted values
    return rmse, mae, rsq, rsqval, adj_rsqval, devexpl, fac2, test_measure_predict


def cross_validate(calib_data, model_var, jobs, feature_cols=FEATURE_COLS,
                   repeat=40, interpolation_factor=0.0):
    """ Cross-validation of GAM """

    # Select test and training dataset for 10 fold cross validation
    kf = KFold(n_splits=10, shuffle=True)

    rmse_model = []
    mae_model = []
    rsq_model = []
    devexpl_model = []
    fac2_model = []
    adj_rsqval_model = []
    rsqval_model = []

    gam_inputs = []
    pool = Pool(processes=int(jobs))

    # Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
    # of the model variables
    for _ in range(repeat):

        # Get the original unshifted grid and split the CV based on this
        og_grid_data = calib_data[(calib_data['y'] %
                                   100 == 0) & (calib_data['x'] % 100 == 0)]

        for train_index_calib, test_index_calib in kf.split(og_grid_data):
            # for train_index_calib, test_index_calib in kf.split(calib_data):
            train_calib_data = calib_data.iloc[train_index_calib]
            test_calib_data = calib_data.iloc[test_index_calib]

            # Gather all cells that do not overlap with a test cell for training
            train_calib_data = calib_data[calib_data.apply(
                lambda c: not is_in(c, test_calib_data), axis=1)]

            # Interpolate new rows for train_calib_data
            train_calib_data = interpolate(train_calib_data, int(
                interpolation_factor * len(train_calib_data))).reset_index()

            # Select test data from model_var (data NOT used for calibration)
            # Do this by finding all rows in model_var whose x and y coordinates are not
            # in train_calib_data
            ind_keys = ['y', 'x']
            ind_train_calib = train_calib_data.set_index(ind_keys).index
            ind_test_calib = test_calib_data.set_index(ind_keys).index
            ind_model_var = model_var.set_index(ind_keys).index

            test_model_var = model_var[~ind_model_var.isin(ind_train_calib)]

            # First gather all the inputs for each GAM calculation in a list
            gam_inputs.append(
                (train_calib_data, test_calib_data, test_model_var, feature_cols))

    try:
        # Add all the GAM calculations with their respective inputs into the Pool
        # returns rmse, rsq, rsqval, devexpl, fac2
        results = pd.DataFrame(pool.map(calculate_gam, gam_inputs))
    except RRuntimeError:
        pool.close()
        print('R Error')
        return {
            'rmse': None,
            'rmse-nomean': None,
            'mae': None,
            'mae-nomean': None,
            'rsq-train': None,
            'rsq': None,
            'rsq-nomean': None,
            'adj-rsq': None,
            'adj-rsq-nomean': None,
            'devexpl': None,
            'fac2-nomean': None,
            'fac2': None,
            'predictions': None}

    pool.close()

    results.columns = ['rmse', 'mae', 'rsq', 'rsqval',
                       'adj-rsqval', 'devexpl', 'fac2', 'predictions']

    # Calculate Root-mean-square error model
    rmse_model.append(results['rmse'])
    mae_model.append(results['mae'])
    # Get R² from summary
    rsq_model.append(results['rsq'])
    devexpl_model.append(results['devexpl'])
    # Calculate Factor of 2
    fac2_model.append(results['fac2'])

    # calculate R2 between predicted and measured concentration
    rsqval_model.append(results['rsqval'])
    adj_rsqval_model.append(results['adj-rsqval'])

    print(results['predictions'][0])
    predictions = pd.concat(results['predictions'].values.tolist())
    predictions = predictions.set_index(['y', 'x'])
    print(predictions)

    skl_rmse = np.sqrt(mean_squared_error(
        predictions['pm_measurement'], predictions['prediction']))
    skl_mae = mean_absolute_error(
        predictions['pm_measurement'], predictions['prediction'])
    skl_rsq_val = r2_score(
        predictions['pm_measurement'], predictions['prediction'])

    # Calculate adjusted R²: rsq-(1-rsq)*p/(n-p-1)
    p = len(feature_cols)
    n = len(calib_data)
    adj_skl_rsq_val = skl_rsq_val - (1 - skl_rsq_val) * p / (n - p - 1)

    # Calculate Factor of 2 metric
    fac2_ind = predictions['pm_measurement'] / \
        predictions['prediction']
    fac2_ind = fac2_ind[(fac2_ind <= 2) & (fac2_ind >= 0.5)].dropna()
    fac2 = (len(fac2_ind) / len(predictions['pm_measurement']) * 100)

    print('Root-mean-square error:', np.mean(rmse_model), 'particles/cm^3')
    print('Mean-absolute error:', np.mean(mae_model))
    print('R2:', np.mean(rsq_model))
    print('R2-val:', np.mean(rsqval_model))
    print('Sklearn R2-val:', skl_rsq_val)
    print('Adjusted Sklearn R2-val:', adj_skl_rsq_val)
    print('DevExpl:', np.mean(devexpl_model) * 100)
    print('FAC2:', np.mean(fac2_model))

    # Metrics with the ending '-nomean' are calculated by appending all labels
    # and predictions of each cross-validation fold to a large list and then
    # calculating the metric based on this large list. The other metrics are
    # calculated for each cross-validation fold and after the cross-validation
    # is done, the mean of these metrics are calculated.
    return {
        'rmse': np.mean(rmse_model),
        'rmse-nomean': skl_rmse,
        'mae': np.mean(mae_model),
        'mae-nomean': skl_mae,
        'rsq-train': np.mean(rsq_model),
        'rsq': np.mean(rsqval_model),
        'rsq-nomean': skl_rsq_val,
        'adj-rsq': np.mean(adj_rsqval_model),
        'adj-rsq-nomean': adj_skl_rsq_val,
        'devexpl': np.mean(devexpl_model) * 100,
        'fac2-nomean': fac2,
        'fac2': np.mean(fac2_model),
        'predictions': predictions}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path', help='File with calibration data',
                        nargs='?', default=INPUT_FILE_PATH)
    parser.add_argument('results_file', default=RESULTS_FILE_NAME, nargs='?',
                        help='File where to output the results')
    parser.add_argument('-mv', '--model_vars', default=MODEL_VAR_PATH,
                        help='Path to model variables')
    parser.add_argument('-f', '--feature_cols', default=FEATURE_COLS,
                        help='Feature columns to use for input')
    parser.add_argument('-j', '--jobs', default=cpu_count(), type=int,
                        help='Specifies the number of simultaneous jobs')
    parser.add_argument('-i', '--interpolation_factor', type=float, default=0.0,
                        help='Number of rows that should be generated through' +
                        ' interpolation as a percentage of train data length ' +
                        '(example: len(train_data) = 200 and -i = 1 -> 200 ' +
                        'interpolated rows, 400 rows overall)')
    args = parser.parse_args()

    # Convert feature cols string to list
    if not isinstance(args.feature_cols, list):
        args.feature_cols = ast.literal_eval(args.feature_cols)

    if not isinstance(args.feature_cols, list):
        print('feature_cols is not a valid list')

    # Load data
    calib_data = load_input_file(args.input_file_path)

    # Load model variables
    model_var = None
    if (args.model_vars.split('.')[-1] == 'mat'):
        # Assume original hasenfratz features when .mat file is choosen
        model_var = pd.DataFrame(sio.loadmat(
            args.model_vars)['model_variables'])
        model_var.columns = ['y', 'x', 'population', 'industry', 'floorlevel',
                             'heating', 'elevation', 'streetsize', 'signaldist',
                             'streetdist', 'slope', 'expo', 'traffic',
                             'streetdist_m', 'streetdist_l', 'trafficdist_l',
                             'trafficdist_h', 'traffic_tot']
        model_var.set_index(['y', 'x'])

    elif (args.model_vars.split('.')[-1] == 'csv'):
        # Assume my own custom features when a csv file is provided
        model_var = pd.read_csv(args.model_vars)
        model_var.set_index(['y', 'x'])

    else:
        print('Unkown model vars file type', args.model_vars.split('.')[-1])

    # Parse timeframe from file name
    tf_pattern = re.compile('\d{8}_\d{8}')
    timeframe = tf_pattern.search(basename(args.input_file_path)).group(0)

    run_info = {
        'source': 'gam',
        'feature_cols': args.feature_cols,
        'tiles': len(calib_data),
        'timeframe': timeframe
    }

    print('Next Run:', run_info)

    # Do 10-fold cross validation on new data set
    results = cross_validate(
        calib_data, model_var, args.jobs, feature_cols=args.feature_cols,
        repeat=1, interpolation_factor=args.interpolation_factor)

    # Merge run information with results
    results = {**run_info, **results}

    # Remove predictions since i can't have a table within a table when
    # outputting to csv
    del results['predictions']

    # Write results to file
    write_results_file(args.results_file, results)
