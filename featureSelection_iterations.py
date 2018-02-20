import argparse
import csv
import pickle

import pandas as pd

import paths
from Models.GAM import GAM
from Models.GAM_featureSelection import GAM_featureSelection
from os.path import join, isfile, isdir
from os import listdir, mkdir

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--seasonNumber", help="Number of season", type=int)
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int)
	parser.add_argument("-p", "--processes", help="Number of parallel processes", type=int)

	args = parser.parse_args()

	seasons = ["pm_ha_ext_01042012_30062012", "pm_ha_ext_01072012_31092012",
	           "pm_ha_ext_01102012_31122012", "pm_ha_ext_01012013_31032013"]

	file = seasons[args.seasonNumber - 1] + '_landUse_withDistances.csv'
	feat = "OSM_land_use_distances"

	feat_columns = ["commercial{}m".format(i) for i in range(50, 3050, 50)]
	feat_columns.extend(["industrial{}m".format(i) for i in range(50, 3050, 50)])
	feat_columns.extend(["residential{}m".format(i) for i in range(50, 3050, 50)])

	feat_columns.extend(["bigStreet{}m".format(i) for i in range(50, 1550, 50)])
	feat_columns.extend(["localStreet{}m".format(i) for i in range(50, 1550, 50)])

	feat_columns.extend(
		["distanceTrafficSignal", "distanceMotorway", "distancePrimaryRoad", "distanceIndustrial"])

	target = 'pm_measurement'

	dir = join(paths.featuresel,
	           "season{}_Features_{}_r2val_{}iterations".format(args.seasonNumber, feat, args.iterations))
	if ~isdir(dir):
		mkdir(dir)

	filelist = listdir(dir)

	filenumber = []
	for f in filelist:
		if isfile(f):
			try:
				filenumber.append(int(f[:-2]))
			except:
				pass

	if len(filenumber) == 0:
		startNumber = 0
	else:
		startNumber = max(filenumber) + 1

	data = []
	with open(paths.lurdata + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	data = pd.DataFrame(data)
	col_total = ['x', 'y', target]
	col_total.extend(feat_columns)
	data.columns = col_total
	print(data.columns)

	dataset = args.seasonNumber

	if args.iterations:
		iterations = args.iterations
	else:
		iterations = 5

	if args.processes:
		njobs = args.processes
	else:
		njobs = 1

	# Feature selection:
	features_total = []
	rmse_total = []
	r2_total = []
	r2val_total = []
	i = 0
	makeIt = True

	for i in range(iterations):
		gam = GAM_featureSelection(njobs=njobs, verbosity=0)
		final_features = gam.select_features(data, feat_columns[:], target)
		gam = GAM(njobs=njobs, niter=iterations, verbosity=1)
		rmse, r2, r2val = gam.test_model(data, final_features, target)

		pickle.dump({'rmse': rmse, 'r2': r2, 'r2val': r2val, 'features': final_features},
		            open(join(dir, "{}.p".format(i)), 'wb'))
