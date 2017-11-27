import sys
from os.path import expanduser
homedir = expanduser("~/")

if (homedir + "code-2017-land-use") not in sys.path:
	print("Adding path to sys.path: " + homedir + "code-2017-land-use")
	sys.path.append(homedir + "code-2017-land-use")

from sklearn.preprocessing import PolynomialFeatures, Imputer, Normalizer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import paths
import numpy as np
import argparse
import csv


def cross_validation(X_t, y_t):
	kf = KFold(n_splits=10, shuffle=True)
	score = []
	score_train = []
	rmse = []
	mae = []

	for train, test in kf.split(X_t):
		X_train, y_train = X_t[train, :], y_t[train]
		X_test, y_test = X_t[test, :], y_t[test]

		imputer = Imputer(strategy='mean')
		scaler = Normalizer()
		preprocessor = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
		regressor = AdaBoostRegressor(n_estimators=489, learning_rate=1.0696244587757953, loss='exponential', )

		line = [('Imputer', imputer), 
			('Scaler', scaler), 
			('Preprocessor', preprocessor), 
			('Regressor', regressor)]
		pipe = Pipeline(line)

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

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help="Define an input file.")
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int)

	args = parser.parse_args()




	if args.file:
		file = args.file
	else:
		file = "pm_ha_ext_01012013_31032013_landUse.csv"
		# file = "pm_ha_ext_01042012_30062012_landUse.csv"
		# file = "pm_ha_ext_01072012_31092012_landUse.csv"
		# file = "pm_ha_ext_01102012_31122012_landUse.csv"


	if args.iterations:
		iterations = args.iterations
	else:
		iterations = 5

	data = []
	with open(paths.lurdata + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	train_data_np = np.array(data)

	X_train, y_train = train_data_np[:, 3:], train_data_np[:, 2]

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
