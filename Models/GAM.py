import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.io as sio
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import paths


class GAM:
	def __init__(self, njobs, niter=40, score="RMSE", verbosity=0):
		self.njobs = njobs
		self.niter = niter
		self.score = score
		self.verbosity = verbosity

	def test_model(self, data, feat_columns, target):

		self.define_formula(feat_columns, target)
		kf = KFold(n_splits=10, shuffle=True)

		rmse_model = []
		rsq_model = []
		devexpl_model = []
		fac2_model = []
		rsqval_model = []

		gam_inputs = []
		pool = Pool(processes=int(self.njobs))

		# Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
		# of the model variables
		for _ in range(self.niter):
			for train_index_calib, test_index_calib in kf.split(data):
				train_calib_data = data.iloc[train_index_calib]
				test_calib_data = data.iloc[test_index_calib]

				# First gather all the inputs for each GAM calculation in a list
				gam_inputs.append((train_calib_data, test_calib_data, self.formula))

		# Add all the GAM calculations with their respective inputs into the Pool
		# returns rmse, rsq, rsqval, devexpl, fac2
		results = pd.DataFrame(pool.map(self.calculate_gam, gam_inputs))

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

		self.print('Root-mean-square error: {} particles/cm^3'.format(np.mean(rmse_model)))
		self.print('R2: {}'.format(np.mean(rsq_model)))
		self.print('R2-val: {}'.format(np.mean(rsqval_model)))
		self.print('DevExpl: {}'.format(np.mean(devexpl_model) * 100))
		self.print('FAC2: {}'.format(np.mean(fac2_model)))

		return np.mean(rmse_model), np.mean(rsq_model), np.mean(rsqval_model)

	def print(self, message):
		if self.verbosity>0:
			print(message)


	def define_formula(self, feat_columns, target):

		# This is the formula for the GAM
		# From https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.terms.html:
		# "s()" defines a smooth term in R
		# "bs" is the basis of the used smooth class
		# "cr" declares a cubic spline basis
		# "k" defines the dimension of the basis (upper limit on degrees of freedom)

		formula = '{}~'.format(target)
		for feature in feat_columns:
			formula += 's({},bs="cr",k=3)+'.format(feature)

		self.formula = formula[:-1]
		return self.formula

	def calculate_gam(self, inputs):

		if len(inputs) == 3:
			train_calib_data, test_calib_data, formula = inputs
		elif len(inputs) == 2:
			train_calib_data, test_calib_data = inputs
			if self.formula:
				formula = self.formula
			else:
				print("Please define formula")
		else:
			print("Wrong number of arguments: Expected 2 or 3, was {}".format(len(inputs)))

		# mgcv is the R package with the GAM implementation
		mgcv = importr('mgcv')
		base = importr('base')
		stats = importr('stats')

		# Activate implicit conversion of pandas to rpy2 and vice versa
		pandas2ri.activate()
		formula = robjects.Formula(formula)

		# Hasenfratz uses a Gamma distribution with a logarithmic link
		family = stats.Gamma(link='log')

		# Train model
		model = mgcv.gam(formula, family, data=train_calib_data)
		su = base.summary(model)

		# Predict the test data
		test_model_var = test_calib_data.drop(['pm_measurement'], axis=1)
		pred_data = stats.predict(model, newdata=test_model_var, type='response')
		test_measure_predict = test_calib_data.assign(prediction=pred_data)

		# Check how large the error is with the remaining 10% of data
		error_model = test_measure_predict['pm_measurement'] - test_measure_predict['prediction']

		# Drop all NaN's
		error_model = error_model.dropna()

		# Calculate Root-mean-square error model
		rmse = np.sqrt(np.mean(error_model ** 2))

		# Get R² from summary
		rsq = su.rx2('r.sq')
		devexpl = su.rx2('dev.expl')

		# Calculate Factor of 2
		fac2_ind = test_measure_predict['pm_measurement'] / test_measure_predict['prediction']
		fac2_ind = fac2_ind[(fac2_ind <= 2) & (fac2_ind >= 0.5)].dropna()
		fac2 = (len(fac2_ind) / len(test_measure_predict['pm_measurement']) * 100)

		# calculate R2 between predicted and measured concentration
		r2val_formula = robjects.Formula('measurements~predictions')
		r2val_env = r2val_formula.environment
		r2val_env['measurements'] = test_measure_predict['pm_measurement']
		r2val_env['predictions'] = test_measure_predict['prediction']
		lt1 = stats.lm(r2val_formula)
		rsqval = base.summary(lt1).rx2('r.squared')

		rsqval = r2_score(test_measure_predict['pm_measurement'], test_measure_predict['prediction'])

		# Return metrics
		return rmse, rsq, rsqval, devexpl, fac2


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-j', '--jobs', default=cpu_count(), type=int,
	                    help='Specifies the number of jobs to run simultaneously.')
	args = parser.parse_args()

	# Load data
	pm_ha = sio.loadmat(paths.extdatadir +
	                    'pm_ha_ext_01042012_30062012.mat')['pm_ha']

	# Prepare data
	data_1 = pd.DataFrame(pm_ha[:, :3])
	data_2 = pd.DataFrame(pm_ha[:, 7:])
	calib_data = pd.concat([data_1, data_2], axis=1)
	calib_data.columns = ["x", "y", "pm_measurement", "population", "industry", "floorlevel", "heating",
	                      "elevation", "streetsize",
	                      "signaldist", "streetdist", "slope", "expo", "traffic", "streetdist_m", "streetdist_l",
	                      "trafficdist_l", "trafficdist_h", "traffic_tot"]

	feat_columns = ['industry', 'floorlevel', 'elevation', 'slope', 'expo', 'streetsize', 'traffic_tot',
	                'streetdist_l']
	target = 'pm_measurement'

	gam = GAM(4, 40)
	gam.test_model(calib_data, feat_columns, target)
