from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import wilcoxon


class RF_featureSelection:
	def __init__(self, njobs, verbosity=0):
		self.njobs = njobs
		self.verbosity = verbosity

	def select_features(self, data, feat_columns, target):
		final_features = []


		r2_check = 0
		r2_old = -1

		treshold = 0.01

		valprev = None
		pvalue = []

		pool = Pool(processes=int(self.njobs))

		while ( ((r2_check - r2_old) > treshold) | (len(final_features) < 2)):

			inputs = []
			for feature in feat_columns:
				features = final_features[:]
				features.append(feature)
				self.print(features)
				inputs.append((data, features, target))

			result = pool.map(self.compute_single, inputs)
			r2_features, rmse_features, feats = list(zip(*result))

			if not valprev:
				valprev = [0]*len(r2_features[0])

			r2_features_final = [np.mean(i) for i in r2_features]
			rmse_features_final = [np.mean(i) for i in rmse_features]

			ind = r2_features_final.index(max(r2_features_final))
			final_features.append(feats[ind])
			feat_columns.pop(feat_columns.index(feats[ind]))

			r2_old = r2_check
			r2_check = r2_features_final[ind]

			pvalue.append(wilcoxon(r2_features[ind], valprev)[1])
			print("R2 value: {:5.3f} \t p-value compared to without last feature: {:5.3f}".format(r2_check, pvalue[-1]))
			print(final_features)
			print()

			valprev = r2_features[ind]

		return final_features, r2_check, rmse_features_final[ind], pvalue

	def compute_single(self, input):

		data, feat_columns, target = input

		kf = KFold(n_splits=10, shuffle=True)

		r2 = []
		rmse = []

		for _ in range(40):
			for train_index_calib, test_index_calib in kf.split(data):
				train_calib_data = data.iloc[train_index_calib]
				test_calib_data = data.iloc[test_index_calib]
				#print(train_calib_data.columns)
				X_train = train_calib_data[feat_columns].values
				X_test = test_calib_data[feat_columns].values
				y_train = train_calib_data[['pm_measurement']].values.ravel()
				y_test = test_calib_data[['pm_measurement']].values.ravel()


				r = RandomForestRegressor()
				r.fit(X_train, y_train)
				pred = r.predict(X_test)
				r2.append(r2_score(pred, y_test))
				rmse.append(np.sqrt(mean_squared_error(pred, y_test)))


		#print(results['rsval'])
		self.print('r2: {}'.format(np.mean(r2)))

		return r2, rmse, feat_columns[-1]

	def print(self, message):
		if self.verbosity > 0:
			print(message)
