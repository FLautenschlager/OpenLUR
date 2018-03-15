"""
Similar to LUR_mapFeatures.py but uses square buffers instead of regular round
buffers. This expects Lv03 (Swiss) coordinates as it is easier to deal with
squares when the coordinates are meters.
"""

import argparse
import csv
import time
import numpy as np
import pandas as pd

import scipy.io as sio
from joblib import Parallel, delayed

from OSM_featureExtraction import Requestor_squares as Requestor
from utils import paths


def preproc_landuse_features_parallel(data, n_workers=1):
	func = preproc_single

	data_new = Parallel(n_jobs=n_workers, verbose=10)(delayed(func)(row) for row in data)
	return data_new


def preproc_single(row):
	r = Requestor.Requestor()

	lat = row[0] + 50
	lon = row[1] + 50
	row_new = [lat, lon]

	row_new.extend(r.create_features(lat, lon))

	row_new[0] -= 50
	row_new[1] -= 50

	return row_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--nWorkers", type=int, default=1, help="Number of parallel processes.")
	parser.add_argument("-s", "--shift", type=int, default=0, help="Meters to shift the original grid")

	args = parser.parse_args()


	bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]

	# Shift bounds
	bounds = [x + args.shift for x in bounds]

	data = []
	for y in range(bounds[0], bounds[1] + 1, 100):
		for x in range(bounds[2], bounds[3] + 1, 100):
			data.append([y, x])
	
	# data = [[684100, 247900]]
	# data = [[681000, 248000]]

	print("Starting feature generation. Total count of cells: {}".format(len(data)))
	start_time = time.time()
	data_new = preproc_landuse_features_parallel(data, n_workers=args.nWorkers)
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = "mapfeatures_zurich_shift_{}.csv".format(str(args.shift))

	data_new = pd.DataFrame(data_new)
	data_new.columns = ['y', 'x', 'roadtype_100', 'roadtype_200', 'roadtype_300', 'roaddist_motorway', 'roaddist_trunk', 'roaddist_primary', 'roaddist_secondary', 'roaddist_tertiary']

	data_new.to_csv(filenew, index=False)

	print("Done! File saved as {}.".format(filenew))
