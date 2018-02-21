import argparse
import csv
import pickle

import pandas as pd
import scipy.io as sio

# from Models.GAM import GAM
from Models.AutoSKLearn import AutoSKLearn
from Models.AutoSKLearn_external import AutoRegressor
from utils import paths
from Models import SGD, RF
from Models.SKlearnModels import *

# from rpy2.rinterface import RRuntimeError

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--seasonNumber", help="Number of season", type=int)
	parser.add_argument("-f", "--features",
	                    help="Dataset to build model on: (1) OpenSense, (2) OSM, (3) OSM + distances", type=int)
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int)
	parser.add_argument("-p", "--processes", help="Number of parallel processes", type=int)
	parser.add_argument("-m", "--model", help="Model: (1) GAM, (2) Auto-sklearn 40*10, (3) Auto-sklearn internal cv", type=int)
	parser.add_argument("-t", "--time", help="Give total time in seconds", type=int)

	args = parser.parse_args()

	seasons = ["pm_ha_ext_01042012_30062012", "pm_ha_ext_01072012_31092012",
	           "pm_ha_ext_01102012_31122012", "pm_ha_ext_01012013_31032013"]

	if args.features == 1:
		file = seasons[args.seasonNumber - 1] + '.mat'
		feat = "OpenSense"
		pm_ha = sio.loadmat(paths.extdatadir + file)['pm_ha']
		# Prepare data
		data_1 = pd.DataFrame(pm_ha[:, :3])
		data_2 = pd.DataFrame(pm_ha[:, 7:])
		data = pd.concat([data_1, data_2], axis=1)
		data.columns = ["x", "y", "pm_measurement", "population", "industry", "floorlevel", "heating",
		                "elevation", "streetsize",
		                "signaldist", "streetdist", "slope", "expo", "traffic", "streetdist_m", "streetdist_l",
		                "trafficdist_l", "trafficdist_h", "traffic_tot"]

		feat_columns = ['industry', 'floorlevel', 'elevation', 'slope', 'expo', 'streetsize', 'traffic_tot',
		                'streetdist_l']
		target = 'pm_measurement'

	elif (args.features == 2) | (args.features == 3):
		file = seasons[args.seasonNumber - 1] + '_landUse.csv'
		feat = "OSM_land_use"

		feat_columns = ["commercial{}m".format(i) for i in range(50, 3050, 50)]
		feat_columns.extend(["industrial{}m".format(i) for i in range(50, 3050, 50)])
		feat_columns.extend(["residential{}m".format(i) for i in range(50, 3050, 50)])

		feat_columns.extend(["bigStreet{}m".format(i) for i in range(50, 1550, 50)])
		feat_columns.extend(["localStreet{}m".format(i) for i in range(50, 1550, 50)])

		target = 'pm_measurement'

		if args.features == 3:
			file = file[:-4] + "_withDistances.csv"
			feat = feat + "_distances"
			feat_columns.extend(
				["distanceTrafficSignal", "distanceMotorway", "distancePrimaryRoad", "distanceIndustrial"])

		data = []
		with open(paths.lurdata + file, 'r') as myfile:
			reader = csv.reader(myfile)
			for row in reader:
				data.append([float(i) for i in row])

		data = pd.DataFrame(data)
		col_total = ['x', 'y']
		col_total.append(target)
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

	if args.model==1:
		print("Install newer R")
		#gam = GAM(njobs=njobs, niter=iterations, verbosity=1)
		#try:
		#	gam.test_model(data, feat_columns, target)
		#except RRuntimeError:
		#	print("Too many features for data")
	elif args.model==2:
		model = AutoRegressor(njobs=njobs, features=feat + "_s{}".format(args.seasonNumber), niter=iterations, verbosity=2, time=args.time)
		model.test_model(data, feat_columns, target)
	elif args.model==3:
		model = AutoSKLearn(njobs=njobs, features=feat + "_s{}".format(args.seasonNumber), time=args.time)
		result = model.test_model(data, feat_columns, target)
		pickle.dump(result, open(paths.autosklearn + "season{}_Features_{}.p".format(args.seasonNumber, feat), 'wb'))
	elif args.model==4:
		print("start SVD")
		model = SGD.SGD(njobs=njobs, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model==5:
		model = RF.RF(njobs=njobs, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)



	# Feature selection:
	#if args.model==3:
		#print("Install newer R")
		#gam = GAM(njobs=njobs, niter=iterations, verbosity=0)

		#final_features = []

		#rmse_check = 10000
		#rmse_old = 20000

		#treshold = 50
		#while (rmse_old-rmse_check > treshold):

		#	rmse_features = []
		#	for feature in feat_columns:
		#		features = final_features[:]
		#		features.append(feature)
		#		print(features)
		#		try:
		#			rmse, r2, r2val = gam.test_model(data, features, target)
		#		except:
		#			break
		#		rmse_features.append(rmse)

		#	print(rmse_features)
		#	ind = rmse_features.index(max(rmse_features))
		#	final_features.append(feat_columns[ind])
		#	feat_columns.pop(ind)
		#	rmse_old = rmse_check
		#	rmse_check, r2, r2val = gam.test_model(data, features, target)
		#	print(rmse_check)
		#	print(final_features)

		#pickle.dump({'rmse':rmse_check, 'r2':r2, 'r2val':r2val, 'features':final_features, 'treshold':treshold, "rmse_old":rmse_old}, open(paths.featuresel + "season{}_Features_{}.p".format(args.seasonNumber, feat), 'wb'))
