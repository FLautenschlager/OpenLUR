import csv
import pickle

import autosklearn.regression
import numpy as np

from utils import paths

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

	X_train, y_train = train_data_np[:,3:], train_data_np[:,2]

	X_train = np.ascontiguousarray(X_train)
	y_train = np.ascontiguousarray(y_train)

	regressor = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=11700, ml_memory_limit=6142, resampling_strategy='cv', resampling_strategy_arguments={'folds':10})
	regressor.fit(X_train, y_train)
	print(regressor.show_models())

	pickle.dump(regressor, open(paths.modeldatadir + "autosklearn_model.p", 'wb'))