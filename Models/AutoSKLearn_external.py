from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from os.path import join
from os import mkdir

# from multiprocessing import Pool
from MyPool import MyPool as Pool
import pickle
import paths
from datetime import datetime


class AutoRegressor:
	def __init__(self, njobs, features, niter=40, verbosity=0, time=60):
		self.njobs = njobs
		self.niter = niter
		self.verbosity = verbosity
		self.groundpath = join(paths.autosklearn, features, datetime.now().strftime("%Y%m%d-%H%M%S"))
		mkdir(self.groundpath)
		self.print(self.groundpath, 1)
		self.time = time

	def test_model(self, data, feat_columns, target):

		kf = KFold(n_splits=10, shuffle=True)

		rmse_model = []
		rsq_model = []

		inputs = []
		pool = Pool(processes=int(self.njobs))
		results = []

		# Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
		# of the model variables
		for i in range(self.niter):
			j = 0
			for train_index_calib, test_index_calib in kf.split(data):
				path = join(self.groundpath, "{}cv{}.p".format(i, j))
				train_calib_data = data.iloc[train_index_calib]
				test_calib_data = data.iloc[test_index_calib]

				# First gather all the inputs for each GAM calculation in a list
				inputs_single = (train_calib_data, test_calib_data, target, feat_columns, path, self.time)
				inputs.append(inputs_single)
				# results.append(self.calculate_AutoSk(inputs_single))
				j += 1

		# Add all the GAM calculations with their respective inputs into the Pool
		# returns rmse, rsq, rsqval, devexpl, fac2
		results = pd.DataFrame(pool.map(self.calculate_AutoSk, inputs))
		pool.close()
		pool.join()
		# results = pd.DataFrame(results)
		results.columns = ['rmse', 'rsq']

		# Calculate Root-mean-square error model
		rmse_model.append(results['rmse'])
		# Get R² from summary
		rsq_model.append(results['rsq'])

		self.print('Mean root-mean-square error: {} particles/cm^3'.format(np.mean(rmse_model)), 1)
		self.print('Mean R2: {}'.format(np.mean(rsq_model)), 1)

		return np.mean(rmse_model), np.mean(rsq_model)

	def print(self, message, verbosity):
		if verbosity <= self.verbosity:
			if verbosity==0:
				print(color.CYAN + message + color.END)
			elif verbosity==1:
				print(color.RED + message + color.END)
			elif verbosity==2:
				print(color.BOLD + message + color.END)



	def calculate_AutoSk(self, inputs):
		train_data, test_data, target, columns, path, time = inputs

		X_train = train_data[columns]
		X_test = test_data[columns]

		y_train = train_data[target]
		y_test = test_data[target]

		automl = AutoSklearnRegressor(time_left_for_this_task=time,
		                              per_run_time_limit=time - 1,
		                              ensemble_size=50,
		                              ensemble_nbest=50,
		                              ml_memory_limit=4096,
		                              resampling_strategy='holdout',
		                              resampling_strategy_arguments={'train_size': 0.8})

		automl.fit(X_train, y_train, metric=mse)

		pred = automl.predict(X_test)

		rmse = np.sqrt(mean_squared_error(y_test, pred))
		r2 = r2_score(y_test, pred)

		self.print('Root-mean-square error: {} particles/cm^3'.format(rmse), 2)
		self.print('R2: {}'.format(r2), 2)

		pickle.dump({'model': automl, 'r2': r2, 'rmse': rmse}, open(path, 'wb'))

		return rmse, r2


class color:
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'
