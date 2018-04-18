import argparse
import csv
import pickle
from os.path import join

import pandas as pd
import scipy.io as sio

from Models.AutoSKLearn import AutoSKLearn
from Models.AutoSKLearn_external import AutoRegressor
from utils import paths
from utils.DataLoader import loadData
from Models.Sklearn_models import RF, RF100, Adaboost, SGD, MLP, GWR, KNN, AutoRF


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--seasonNumber", help="Number of season", type=int)
	parser.add_argument("-f", "--features",
	                    help="Dataset to build model on: (1) OpenSense, (2) OSM, (3) OSM + distances", type=int)
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int)
	parser.add_argument("-p", "--processes", help="Number of parallel processes", type=int)
	parser.add_argument("-m", "--model", help="Model: (GAM) GAM, (AML) Auto-sklearn 40*10, (AMLlong) Auto-sklearn internal cv", type=str)
	parser.add_argument("-t", "--time", help="Give total time in seconds", type=int)

	args = parser.parse_args()

	data, feat_columns, target, feat = loadData(args.seasonNumber, args.features)

	dataset = args.seasonNumber

	if args.iterations:
		iterations = args.iterations
	else:
		iterations = 5

	if args.processes:
		njobs = args.processes
	else:
		njobs = 1

	outpath = join(str(args.seasonNumber), feat + "_" + str(args.model))

	if args.model=="GAM":
		try:
			from Models.GAM import GAM
			from rpy2.rinterface import RRuntimeError
			gam = GAM(njobs=njobs, niter=iterations, verbosity=1, name=outpath)
			if (feat == "OSM_land_use_distances"):
				feat_columns = ["residential1100m", "distanceTrafficSignal", "distanceMotorway", "residential2000m",
								"residential1950m", "residential1300m", "residential1850m", "industrial1850m",
								"industrial2300m", "commercial2100m", "bigStreet100m", "bigStreet50m", "bigStreet200m",
								"industrial2950m", "industrial2450m", "industrial1700m"]

			try:
				gam.test_model(data, feat_columns, target)
			except RRuntimeError:
				print("Too many features for data")
		except Exception as e:
			print(e)
			print("Install newer R")


	elif args.model=="AML":
		model = AutoRegressor(njobs=njobs, features=feat + "_s{}".format(args.seasonNumber), niter=iterations, verbosity=2, time=args.time)
		model.test_model(data, feat_columns, target)
	elif args.model=="AMLlong":
		model = AutoSKLearn(njobs=njobs, features=feat + "_s{}".format(args.seasonNumber), time=args.time)
		result = model.test_model(data, feat_columns, target)
		pickle.dump(result, open(paths.autosklearn + "season{}_Features_{}.p".format(args.seasonNumber, feat), 'wb'))
	elif args.model=="SGD":
		model = SGD(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model=="RF":
		model = RF(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model == "RF100":
		model = RF100(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model=="KNN":
		model = KNN(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model=="ADA":
		model = Adaboost(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model == "MLP":
		model = MLP(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model == "AutoRF":
		model = AutoRF(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
	elif args.model=="GWR":
		if (feat=="OSM_land_use_distances"):
			feat_columns = ["residential1100m", "distanceTrafficSignal", "distanceMotorway", "residential2000m", "residential1950m", "residential1300m", "residential1850m", "industrial1850m", "industrial2300m", "commercial2100m", "bigStreet100m", "bigStreet50m", "bigStreet200m", "industrial2950m", "industrial2450m", "industrial1700m"]
		print(feat_columns)
		model = GWR(njobs=njobs, name=outpath, niter=iterations, verbosity=2)
		model.test_model(data, feat_columns, target)
