import sys

if "/home/florian/Code/code-2017-land-use" not in sys.path:
	sys.path.append("/home/florian/Code/code-2017-land-use")

from sklearn.preprocessing import PolynomialFeatures, Imputer, StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import paths
import numpy as np
import scipy.io as sio


def cross_validation(X_t, y_t):
	kf = KFold(n_splits=10, shuffle=True)
	score = []
	score_train = []
	rmse = []
	mae = []

	for train, test in kf.split(X_t):
		X_train, y_train = X_t[train, :], y_t[train]
		X_test, y_test = X_t[test, :], y_t[test]

		imputer = Imputer(strategy='most_frequent')
		scaler = StandardScaler()
		preprocessor = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
		regressor = AdaBoostRegressor(n_estimators=295, learning_rate=0.15423925490619383, loss='linear', )

		pipe = Pipeline(
			[('Imputer', imputer), ('Scaler', scaler), ('Polynomial', preprocessor), ('Regressor', regressor)])

		pipe.fit(X_train, y_train)

		pred = pipe.predict(X_test)

		score_train.append(r2_score(y_train, pipe.predict(X_train)))
		score.append(r2_score(y_test, pred))
		mae.append(mean_absolute_error(y_test, pred))
		rmse.append(mean_squared_error(y_test, pred))

	scoreMean = np.mean(score)
	# print("R2-score on training folds = " + score_train)
	# print("R2-score on test folds = " + score)
	print("Mean R2-score on training data = {}".format(np.mean(score_train)))
	print("Mean R2-score on test data = {}".format(np.mean(scoreMean)))
	return scoreMean, np.mean(mae), np.sqrt(np.mean(rmse))


if __name__ == "__main__":

	file = "pm_ha_ext_01012013_31032013.mat"
	# file = "pm_ha_ext_01042012_30062012.mat"
	# file = "pm_ha_ext_01072012_31092012.mat"
	# file = "pm_ha_ext_01102012_31122012.mat"

	iterations = 40

	data = []
	data = sio.loadmat(paths.extdatadir + 'pm_ha_ext_01042012_30062012.mat')['pm_ha']

	train_data_np = np.array(data)

	X_train, y_train = train_data_np[:, 7:], train_data_np[:, 2]

	X_train = np.ascontiguousarray(X_train)
	y_train = np.ascontiguousarray(y_train)

	r2_total = []
	mae_total = []
	rmse_total = []
	for _ in range(iterations):
		r2, mae, rmse = cross_validation(X_train, y_train)
		r2_total.append(r2)
		mae_total.append(mae)
		rmse_total.append(rmse)

	print("R2 = {}\nMAE = {}\nRMSE = {}".format(np.mean(r2_total), np.mean(mae_total), np.mean(rmse_total)))
