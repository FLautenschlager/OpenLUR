# -*- encoding: utf-8 -*-
import multiprocessing
import shutil
import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.metrics import r2, mean_squared_error
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.constants import *
from os.path import join, isdir
from os import mkdir

from datetime import datetime


class AutoSKLearn:

	def __init__(self, njobs, features, time=60):

		self.njobs = njobs
		self.dir = join('tmp', features + datetime.now().strftime("%Y%m%d-%H%M%S"))
		if ~isdir(self.dir):
			mkdir(self.dir)
		self.tmp_folder = join(self.dir, 'tmp')
		self.output_folder = join(self.dir, 'out')
		if ~isdir(self.tmp_folder):
			mkdir(self.tmp_folder)
		if ~isdir(self.output_folder):
			mkdir(self.output_folder)

		self.time = time

		for dir in [self.tmp_folder, self.output_folder]:
			try:
				shutil.rmtree(dir)
			except OSError as e:
				pass

	def test_model(self, data, feat_columns, target):

		X_train = data[feat_columns].values
		y_train = data[target].values


		processes = []
		spawn_regressor = self.get_spawn_regressor(X_train, y_train)
		for i in range(4):  # set this at roughly half of your cores
			p = multiprocessing.Process(target=spawn_regressor, args=(i, 'LUR'))
			p.start()
			processes.append(p)
		for p in processes:
			p.join()

		print('Starting to build final model!')
		automl = AutoSklearnRegressor(
			time_left_for_this_task=61,  # sec., how long should this seed fit process run
			per_run_time_limit=60,  # sec., each model may only take this long before it's killed
			ml_memory_limit=1024 * 4,  # MB, memory limit imposed on each call to a ML algorithm
			shared_mode=True,  # tmp folder will be shared between seeds
			tmp_folder=self.tmp_folder,
			output_folder=self.output_folder,
			delete_tmp_folder_after_terminate=False,
			delete_output_folder_after_terminate=False,
			ensemble_size=20,  # ensembles will be built when all optimization runs are finished
			initial_configurations_via_metalearning=0,
			seed=0,
			resampling_strategy='cv',
			resampling_strategy_arguments={'folds': 10}
		)
		automl.fit(X_train, y_train, dataset_name='LUR')
		automl.fit_ensemble(y_train, task=REGRESSION, metric=mean_squared_error, dataset_name='LUR', ensemble_nbest=20, ensemble_size=20)
		#predictions = automl.predict(X_train)

		#print(automl.show_models())
		#print("Accuracy score", np.sqrt(mean_squared_error(y_train, predictions)))
		#print(automl)
		models = automl.get_models_with_weights()
		print(models[0][0])
		print(models[0][1])
		return models

	def get_spawn_regressor(self, X_train, y_train):
		def spawn_regressor(seed, dataset_name):
			"""Spawn a subprocess.
			auto-sklearn does not take care of spawning worker processes. This
			function, which is called several times in the main block is a new
			process which runs one instance of auto-sklearn.
			"""

			# Use the initial configurations from meta-learning only in one out of
			# the four processes spawned. This prevents auto-sklearn from evaluating
			# the same configurations in four processes.
			if seed == 0:
				initial_configurations_via_metalearning = 25
				smac_scenario_args = {}
			else:
				initial_configurations_via_metalearning = 0
				smac_scenario_args = {'initial_incumbent': 'RANDOM'}

			# Arguments which are different to other runs of auto-sklearn:
			# 1. all classifiers write to the same output directory
			# 2. shared_mode is set to True, this enables sharing of data between
			# models.
			# 3. all instances of the AutoSklearnClassifier must have a different seed!
			automl = AutoSklearnRegressor(
				time_left_for_this_task=self.time,  # sec., how long should this seed fit process run
				per_run_time_limit=int(self.time/10),  # sec., each model may only take this long before it's killed
				ml_memory_limit=1024*4,  # MB, memory limit imposed on each call to a ML algorithm
				shared_mode=True,  # tmp folder will be shared between seeds
				tmp_folder=self.tmp_folder,
				output_folder=self.output_folder,
				delete_tmp_folder_after_terminate=False,
				delete_output_folder_after_terminate=False,
				ensemble_size=0,  # ensembles will be built when all optimization runs are finished
				initial_configurations_via_metalearning=initial_configurations_via_metalearning,
				seed=seed,
				resampling_strategy='cv',
				resampling_strategy_arguments={'folds':10},
				smac_scenario_args=smac_scenario_args
			)
			automl.fit(X_train, y_train, dataset_name=dataset_name)


		return spawn_regressor

