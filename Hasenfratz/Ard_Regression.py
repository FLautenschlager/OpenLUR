import sys
if "/home/florian/Code/code-2017-land-use" not in sys.path:
        sys.path.append("/home/florian/Code/code-2017-land-use")

from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from utils import paths
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

		im = Imputer(strategy='most_frequent')
		mm = MinMaxScaler()
		p = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
		r = ARDRegression(n_iter=300, tol=8.567911551771192e-05, alpha_1=0.0008413297779810135,
		                  alpha_2=1.8746703201245091e-06, lambda_1=5.453277023445055e-07, lambda_2=2.26448162723143e-07,
		                  threshold_lambda=1900.8975967383842, fit_intercept=True)

		pipe = Pipeline([('Imputer', im), ('Scaler', mm), ('Polynomial', p), ('Regressor', r)])

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
