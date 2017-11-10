import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import paths
from multiprocessing import Pool, cpu_count
import argparse
import pickle

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l']

def build_formula_string(feature_cols):
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
        test_model_var_predictions[['x', 'y', 'prediction']], how='inner', on=['x', 'y']
        )#[['x', 'y', 'pm_measurement', 'prediction']]
    # Check how large the error is with the remaining 10% of data
    error_model = test_measure_predict['pm_measurement'] - \
        test_measure_predict['prediction']
    # Drop all NaN's
    error_model = error_model.dropna()
    # Calculate Root-mean-square error model
    rmse = np.sqrt(np.mean(error_model**2))
    # Get R² from summary
    rsq = su.rx2('r.sq')
    devexpl = su.rx2('dev.expl')
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
    rsqval = base.summary(lt1).rx2('r.squared')

    # Return metrics and predicted values
    return rmse, rsq, rsqval, devexpl, fac2, test_measure_predict

def cross_validate(calib_data, model_var, jobs, feature_cols=FEATURE_COLS, repeat=40,):
    # Select test and training dataset for 10 fold cross validation
    kf = KFold(n_splits=10, shuffle=True)

    rmse_model = []
    rsq_model = []
    devexpl_model = []
    fac2_model = []
    rsqval_model = []

    gam_inputs = []
    pool = Pool(processes=int(jobs))

    # Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
    # of the model variables
    for _ in range(repeat):
        for train_index_calib, test_index_calib in kf.split(calib_data):
            train_calib_data = calib_data.iloc[train_index_calib]
            test_calib_data = calib_data.iloc[test_index_calib]

            # Select test data from model_var (data NOT used for calibration)
            # Do this by finding all rows in model_var whose x and y coordinates are not
            # in train_calib_data
            ind_keys = ['x', 'y']
            ind_train_calib = train_calib_data.set_index(ind_keys).index
            ind_test_calib = test_calib_data.set_index(ind_keys).index
            ind_model_var = model_var.set_index(ind_keys).index

            test_model_var = model_var[~ind_model_var.isin(ind_train_calib)]

            # First gather all the inputs for each GAM calculation in a list
            gam_inputs.append((train_calib_data, test_calib_data, test_model_var, feature_cols))


    try:
        # Add all the GAM calculations with their respective inputs into the Pool
        # returns rmse, rsq, rsqval, devexpl, fac2
        results = pd.DataFrame(pool.map(calculate_gam, gam_inputs))
    except RRuntimeError:
        pool.close()
        print('R Error')
        return {
            'mean-batch-rmse': None,
            'rmse': None,
            'mean-batch-rsq': None,
            'mean-batch-rsq-val': None,
            'rsq-val': None,
            'adj-rsq-val': None,
            'devexpl': None,
            'fac2': None,
            'predictions': None}


    pool.close()

    results.columns = ['rmse', 'rsq', 'rsqval', 'devexpl', 'fac2', 'predictions']

    # Calculate Root-mean-square error model
    rmse_model.append(results['rmse'])
    # Get R² from summary
    rsq_model.append(results['rsq'])
    devexpl_model.append(results['devexpl'])
    # Calculate Factor of 2
    fac2_model.append(results['fac2'])

    # calculate R2 between predicted and measured concentration
    rsqval_model.append(results['rsqval'])

    print(results['predictions'][0])
    predictions = pd.concat(results['predictions'].values.tolist())
    predictions = predictions.set_index(['x', 'y'])
    print(predictions)

    pickle.dump(predictions, open('hasenfratz_results.p', 'wb'))

    skl_rmse = np.sqrt(mean_squared_error(predictions['pm_measurement'], predictions['prediction']))
    skl_rsq_val = r2_score(predictions['pm_measurement'], predictions['prediction'])

    # Calculate adjusted R²: rsq-(1-rsq)*p/(n-p-1)
    p = len(feature_cols)
    n = len(calib_data)
    adj_skl_rsq_val = skl_rsq_val-(1-skl_rsq_val)*p/(n-p-1)


    print('Root-mean-square error:', np.mean(rmse_model), 'particles/cm^3')
    print('R2:', np.mean(rsq_model))
    print('R2-val:', np.mean(rsqval_model))
    print('Sklearn R2-val:', skl_rsq_val)
    print('Adjusted Sklearn R2-val:', adj_skl_rsq_val)
    print('DevExpl:', np.mean(devexpl_model) * 100)
    print('FAC2:', np.mean(fac2_model))

    return {
        'mean-batch-rmse': np.mean(rmse_model),
        'rmse': skl_rmse,
        'mean-batch-rsq': np.mean(rsq_model),
        'mean-batch-rsq-val': np.mean(rsqval_model),
        'rsq-val': skl_rsq_val,
        'adj-rsq-val': adj_skl_rsq_val,
        'devexpl': np.mean(devexpl_model) * 100,
        'fac2': np.mean(fac2_model),
        'predictions': predictions}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', default=cpu_count(), type=int, help='Specifies the number of jobs to run simultaneously.')
    parser.add_argument('-c', '--cross', action='store_true',
                        help='Train and evaluate with 10-fold cross validation.')
    parser.add_argument('-b', '--batch_cross', action='store_true',
                        help='Train and evaluate batches with 10-fold cross validation.')
    args = parser.parse_args()

    if args.cross is True:
        # Load data
        pm_ha = sio.loadmat(paths.extdatadir +
                            'pm_ha_ext_01102012_31122012.mat')['pm_ha']#'pm_ha_ext_01042012_30062012.mat')['pm_ha']

        # Prepare data
        data_1 = pd.DataFrame(pm_ha[:, :3])
        data_2 = pd.DataFrame(pm_ha[:, 7:])
        calib_data = pd.concat([data_1, data_2], axis=1)
        calib_data.columns = ['x', 'y', 'pm_measurement', 'population', 'industry', 'floorlevel', 'heating', 'elevation', 'streetsize',
                              'signaldist', 'streetdist', 'slope', 'expo', 'traffic', 'streetdist_m', 'streetdist_l', 'trafficdist_l', 'trafficdist_h', 'traffic_tot']

        model_var = pd.DataFrame(sio.loadmat(
            paths.rdir + 'model_ha_variables.mat')['model_variables'])
        model_var.columns = ['x', 'y', 'population', 'industry', 'floorlevel', 'heating', 'elevation', 'streetsize', 'signaldist',
                             'streetdist', 'slope', 'expo', 'traffic', 'streetdist_m', 'streetdist_l', 'trafficdist_l', 'trafficdist_h', 'traffic_tot']

        cross_validate(calib_data, model_var, args.jobs)

    if args.batch_cross is True:
        calib_files = [
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_1200.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_1100.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_1000.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_900.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_800.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_700.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_600.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_500.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_400.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_300.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_250.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_200.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_170.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_140.csv',
            paths.extdatadir + 'pm_01072012_31092012_filtered_ha_110.csv'#,
            # paths.extdatadir + 'pm_01072012_31092012_filtered_ha_80.csv',
            # paths.extdatadir + 'pm_01072012_31092012_filtered_ha_50.csv',
            # paths.extdatadir + 'pm_01072012_31092012_filtered_ha_20.csv'
        ]
        feature_cols_list = [
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population',
            #     'heating'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population',
            #     'heating', 'signaldist'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population',
            #     'heating', 'signaldist', 'streetdist'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population',
            #     'heating', 'signaldist', 'streetdist', 'traffic'],
            # ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
            #     'streetsize', 'traffic_tot', 'streetdist_l', 'population',
            #     'heating', 'signaldist', 'streetdist', 'traffic',
            #     'streetdist_m'],
            ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l', 'population',
                'heating', 'signaldist', 'streetdist', 'traffic',
                'streetdist_m', 'trafficdist_l'],
            ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l', 'population',
                'heating', 'signaldist', 'streetdist', 'traffic',
                'streetdist_m', 'trafficdist_l', 'trafficdist_h']
        ]

        results_list = []

        for feature_cols in feature_cols_list:
            for calib_file in calib_files:
                # Load new data set from file
                calib_data = pd.read_csv(calib_file)

                # Load model_var
                model_var = pd.DataFrame(sio.loadmat(
                    paths.rdir + 'model_ha_variables.mat')['model_variables'])
                model_var.columns = ['x', 'y', 'population', 'industry', 'floorlevel', 'heating', 'elevation', 'streetsize', 'signaldist',
                                     'streetdist', 'slope', 'expo', 'traffic', 'streetdist_m', 'streetdist_l', 'trafficdist_l', 'trafficdist_h', 'traffic_tot']

                run_info = {
                    'source': 'Hasenfratz',
                    'feature_cols': feature_cols,
                    'tiles': len(calib_data),
                    'timeframe': '01072012_31092012'
                }

                print('Next CV:', run_info)

                # Do 10-fold cross validation on new data set
                results = cross_validate(calib_data, model_var, args.jobs, feature_cols=feature_cols, repeat=1)

                # Merge run information with results
                results = {**run_info, **results}
                # Calculate adjusted R²: rsq-(1-rsq)*p/(n-p-1)
                # rsq = results['rsq-val']
                # p = len(feature_cols)
                # n = results['tiles']
                # results['ajd-rsq-val'] = rsq-(1-rsq)*p/(n-p-1)
                # Remove predictions since i can't have a table in a table when
                # outputting to csv
                del results['predictions']

                # Store results in a list
                results_list.append(results)

        results_list_df = pd.DataFrame(results_list)
        results_list_df.to_csv('tiles_features_batchcv_hasenfratz.csv', mode='a', index=False)
