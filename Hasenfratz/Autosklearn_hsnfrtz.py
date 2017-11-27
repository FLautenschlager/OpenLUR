import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
import paths
import csv
import numpy as np
import pickle
import scipy.io as sio


if __name__ == "__main__":

	file = "pm_ha_ext_01012013_31032013.mat"
	# file = "pm_ha_ext_01042012_30062012.mat"
	# file = "pm_ha_ext_01072012_31092012.mat"
	# file = "pm_ha_ext_01102012_31122012.mat"

	data = []
	data = sio.loadmat(paths.extdatadir + 'pm_ha_ext_01042012_30062012.mat')['pm_ha']

	train_data_np = np.array(data)

	X_train, y_train = train_data_np[:,7:], train_data_np[:,2]

	X_train = np.ascontiguousarray(X_train)
	y_train = np.ascontiguousarray(y_train)

	regressor = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=11700, ml_memory_limit=6142, resampling_strategy='cv', resampling_strategy_arguments={'folds':10})
	regressor.fit(X_train, y_train)
	print(regressor.show_models())

	pickle.dump(regressor, open(paths.modeldatadir + file[:-4] +  "_autosklearn_Hsnfrtz_model.p", 'wb'))
