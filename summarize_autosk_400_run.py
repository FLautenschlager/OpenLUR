import argparse
import csv
import pickle

import pandas as pd

from utils.MyPool import MyPool

from os import listdir, mkdir
from os.path import join, isfile, isdir


# from rpy2.rinterface import RRuntimeError


def process_single(file):
	data = pickle.load(open(file, "rb"))
	rmse = data['rmse']
	r2 = data['r2']

	if 'best_model' in data.keys():
		model = data['best_model']
	else:
		model = data['model'].get_models_with_weights()[0][1].configuration.get_dictionary()['regressor:__choice__']

	return rmse, r2, model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--path", help="Path of model .p", type=str)
	parser.add_argument("-n", "--numberWorkers", help="Number of parallel workers used", type=int)


	args = parser.parse_args()

	path = args.path.split("/")
	outpath = ""
	for p in path[-1]:
		outpath = join(outpath, p)

	filelist = listdir(str(path))

	input = []
	for file in filelist:
		input.append(join(args.path, file))

	pool = MyPool(processes=args.numberWorkers)
	results = pd.DataFrame(pool.map(process_single, input))
	pool.close()
	pool.join()

	results.columns = ["rmse", "r2", "model"]

	results.to_csv(outpath + ".csv")



