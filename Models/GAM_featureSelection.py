from multiprocessing import Pool

import pandas as pd
from sklearn.model_selection import KFold

from Models.GAM import GAM


class GAM_featureSelection:
	def __init__(self, njobs, verbosity=0):
		self.njobs = njobs
		self.verbosity = verbosity

	def select_features(self, data, feat_columns, target):
		final_features = []


		r2_check = 0
		r2_old = -1

		treshold = 0.01

		pool = Pool(processes=int(self.njobs))

		while ( r2_check - r2_old) > treshold:

			inputs = []
			for feature in feat_columns:
				features = final_features[:]
				features.append(feature)
				self.print(features)
				inputs.append((data, features, target))

			result = pool.map(self.compute_single, inputs)
			r2_features, rmse_features, feats = list(zip(*result))

			r2_features_final = [np.mean(i) for i in r2_features]
			rmse_features_final = [np.mean(i) for i in rmse_features]

			ind = r2_features_final.index(max(r2_features_final))
			final_features.append(feats[ind])
			feat_columns.pop(feat_columns.index(feats[ind]))
			r2_old = r2_check
			r2_check = r2_features_final[ind]
			rmse_check = rmse_features_final[ind]
			print(r2_check)
			print(final_features)

		return final_features, r2_check, rmse_check

	def compute_single(self, input):

		data, feat_columns, target = input

		kf = KFold(n_splits=10, shuffle=True)

		results = []
		gam = GAM(1)

		for _ in range(5):
			for train_index_calib, test_index_calib in kf.split(data):
				train_calib_data = data.iloc[train_index_calib]
				test_calib_data = data.iloc[test_index_calib]

				# First gather all the inputs for each GAM calculation in a list
				gam_inputs = (train_calib_data, test_calib_data)
				gam.define_formula(feat_columns, target)
				results.append(gam.calculate_gam(gam_inputs))

		# Add all the GAM calculations with their respective inputs into the Pool
		# returns rmse, rsq, rsqval, devexpl, fac2
		results = pd.DataFrame(results)

		results.columns = ['rmse', 'rsq']

		# Calculate Root-mean-square error model
		rmse_model = results['rmse'].mean()
		r2val = results['rsq'].mean()
		list(r2 = results['rmse'].values)
		list(rmse = results['rsq'].values)
		#print(results['rsval'])
		self.print('Root-mean-square error: {} particles/cm^3'.format(rmse_model))

		return r2, rmse, feat_columns[-1]

	def print(self, message):
		if self.verbosity > 0:
			print(message)
