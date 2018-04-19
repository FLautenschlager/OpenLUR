import pickle
from os import listdir, mkdir
from os.path import join, isfile, isdir

import numpy as np
import pandas as pd
from autosklearn.metrics import mean_squared_error as mse
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

from utils import paths
# from multiprocessing import Pool
from utils.MyPool import MyPool as Pool
from utils.color import Color


class AutoRegressor:
	def __init__(self, njobs, features, niter=40, verbosity=0, time=60):
		self.njobs = njobs
		self.niter = niter
		self.verbosity = verbosity
		self.groundpath = join(paths.autosklearn, features, "time{}s".format(time))
		#print(not isdir(self.groundpath))
		if not isdir(self.groundpath):
			mkdir(self.groundpath)
		self.print(self.groundpath, 1)
		self.time = time

	def test_model(self, data, feat_columns, target):

		kf = KFold(n_splits=10, shuffle=True)

		inputs = []
		filelist = listdir(self.groundpath)

		for f in filelist:
			self.print(f, 1)

		# Hasenfratz does the 10 fold cross validation 40 times to get a better coverage
		# of the model variables
		for i in range(self.niter):

			makeIt = False
			for j in range(10):
				file = "{}cv{}.p".format(i, j)
				if not (file in filelist):
					makeIt = True

			j = 0

			if makeIt:
				for train_index_calib, test_index_calib in kf.split(data):
					iterationInput = []
					file = "{}cv{}.p".format(i, j)
					path = join(self.groundpath, file)
					train_calib_data = data.iloc[train_index_calib]
					test_calib_data = data.iloc[test_index_calib]

					# First gather all the inputs for each GAM calculation in a list
					inputs_single = (train_calib_data, test_calib_data, target, feat_columns, path, self.time)
					iterationInput.append(inputs_single)
					inputs.append(inputs_single)
					# results.append(self.calculate_AutoSk(inputs_single))
					j += 1

		# Compute in parallel
		pool = Pool(processes=int(self.njobs))
		pool.map(self.calculate_AutoSk, inputs)
		pool.close()
		pool.join()

		# Load all results
		filelist = listdir(self.groundpath)
		rmse = []
		r2 = []

		for f in filelist:

			try:
				data = pickle.load(open(join(self.groundpath, f), "rb"))
				rmse.append(data['rmse'])
				r2.append(data['r2'])
			except:
				pass

		# Calculate Root-mean-square error model
		rmse_model = np.mean(rmse)
		# Get R² from summary
		rsq_model = np.mean(r2)

		self.print('Mean root-mean-square error: {} particles/cm^3'.format(rmse_model), 1)
		self.print('Mean R2: {}'.format(rsq_model), 1)

		return rmse_model, rsq_model, self.groundpath

	def print(self, message, verbosity):
		if verbosity <= self.verbosity:
			if verbosity==0:
				print(Color.CYAN + message + Color.END)
			elif verbosity==1:
				print(Color.RED + message + Color.END)
			elif verbosity==2:
				print(Color.BOLD + message + Color.END)



	def calculate_AutoSk(self, inputs):
		train_data, test_data, target, columns, path, time = inputs

		X_train = train_data[columns]
		X_test = test_data[columns]

		y_train = train_data[target]
		y_test = test_data[target]

		# self.print("Doing {}".format(path), 1)

		automl = AutoSklearnRegressor(time_left_for_this_task=time,
		                              per_run_time_limit=time - 3,
		                              ensemble_size=50,
		                              ensemble_nbest=50,
		                              ml_memory_limit=4096,
		                              resampling_strategy='holdout',
		                              resampling_strategy_arguments={'train_size': 0.8})

		automl.fit(X_train, y_train, metric=mse)
		automl.refit(X_train, y_train)

		pred = automl.predict(X_test)

		rmse = np.sqrt(mean_squared_error(y_test, pred))
		r2 = r2_score(y_test, pred)

		self.print('Run {}'.format(path[-7:-2]), 2)
		self.print('Root-mean-square error: {} particles/cm^3'.format(rmse), 2)
		self.print('R2: {}'.format(r2), 2)

		pickle.dump({'best_model': automl.get_models_with_weights(), 'r2': r2, 'rmse': rmse}, open(path, 'wb'))

		return rmse, r2

