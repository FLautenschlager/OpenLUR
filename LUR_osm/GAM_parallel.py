import csv
# from LUR_preprocessing import query_osm_polygone, query_osm_highway
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import KFold

from utils import paths


def calculate_gam(inputs):
	train_calib_data, test_calib_data, test_model_var = inputs

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

	form = '2~s(3,bs="cr",k=3)'

	for i in range(4, 243):
		 form += '+s({}.bs="cr",k=3)'.format(i)

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
		test_model_var_predictions, how='inner', on=['x', 'y']
	)  # [['x', 'y', 'pm_measurement', 'prediction']]
	# Check how large the error is with the remaining 10% of data
	error_model = test_measure_predict['pm_measurement'] - \
	              test_measure_predict['prediction']
	# Drop all NaN's
	error_model = error_model.dropna()
	# Calculate Root-mean-square error model
	rmse = np.sqrt(np.mean(error_model ** 2))
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

	# Return metrics
	return rmse, rsq, rsqval, devexpl, fac2


if __name__ == "__main__":

	file = "pm_ha_ext_01012013_31032013_landUse.csv"
	# file = "pm_ha_ext_01042012_30062012_landUse.csv"
	# file = "pm_ha_ext_01072012_31092012_landUse.csv"
	# file = "pm_ha_ext_01102012_31122012_landUse.csv"

	data = []
	with open(paths.lurdata + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	train_data_np = np.array(data)

	kf = KFold(n_splits=10, shuffle=True)

	rmse_model = []
	rsq_model = []
	devexpl_model = []
	fac2_model = []
	rsqval_model = []

	gam_inputs = []
	pool = Pool(processes=4)

	# Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
	# of the model variables
	for _ in range(40):
		for train_index_calib, test_index_calib in kf.split(train_data_np):
			train_calib_data = train_data_np[train_index_calib, :]
			test_calib_data = train_data_np[test_index_calib, :]

			# Select test data from model_var (data NOT used for calibration)
			# Do this by finding all rows in model_var whose x and y coordinates are not
			# in train_calib_data
			ind_keys = ['x', 'y']
			ind_train_calib = train_calib_data.set_index(ind_keys).index
			ind_test_calib = test_calib_data.set_index(ind_keys).index
			ind_model_var = model_var.set_index(ind_keys).index

			test_model_var = model_var[~ind_model_var.isin(ind_train_calib)]

			# First gather all the inputs for each GAM calculation in a list
			gam_inputs.append((train_calib_data, test_calib_data, test_model_var))

	# Add all the GAM calculations with their respective inputs into the Pool
	# returns rmse, rsq, rsqval, devexpl, fac2
	results = pd.DataFrame(pool.map(calculate_gam, gam_inputs))

	results.columns = ['rmse', 'rsq', 'rsqval', 'devexpl', 'fac2']

	# Calculate Root-mean-square error model
	rmse_model.append(results['rmse'])
	# Get R² from summary
	rsq_model.append(results['rsq'])
	devexpl_model.append(results['devexpl'])
	# Calculate Factor of 2
	fac2_model.append(results['fac2'])

	# calculate R2 between predicted and measured concentration
	rsqval_model.append(results['rsqval'])

	print('Root-mean-square error:', np.mean(rmse_model), 'particles/cm^3')
	print('R2:', np.mean(rsq_model))
	print('R2-val:', np.mean(rsqval_model))
	print('DevExpl:', np.mean(devexpl_model) * 100)
	print('FAC2:', np.mean(fac2_model))
