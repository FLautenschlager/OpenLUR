import argparse
import csv
import time

import scipy.io as sio
from joblib import Parallel, delayed

from OSM_featureExtraction import Requestor
from utils import paths
from utils.wgs84_ch1903 import *


def create_features_from_SwissCoord(x, y):
	lat = CHtoWGSlat(x, y)
	lon = CHtoWGSlng(x, y)
	return r.create_features(lon, lat)


def preproc_landuse_features(data):
	if args.distance:
		func = preproc_single_with_dist
	else:
		func = preproc_single

	data_new = []
	total_len = len(data)
	for i, row in enumerate(data):

		data_new.append(func(row))

		if i % 100 == 0:
			print("{}".format(float(i) / total_len))

	return data_new

def preproc_landuse_features_parallel(data, n_workers=1):
	if args.distance:
		func = preproc_single_with_dist
	else:
		func = preproc_single

	data_new = Parallel(n_jobs=n_workers, verbose=10)(delayed(func)(row) for row in data)
	return data_new

def preproc_single(row):
	r = Requestor.Requestor("zurich")
	x = row[0]
	y = row[1]
	row_new = [x, y, 0]
	lat = CHtoWGSlat(x + 50, y + 50)
	lon = CHtoWGSlng(x + 50, y + 50)

	row_new.extend(r.create_features(lon, lat))

	return row_new

def preproc_single_with_dist(row):
	r = Requestor.Requestor("zurich")
	x = row[0]
	y = row[1]
	row_new = [x, y, 0]
	lat = CHtoWGSlat(x + 50, y + 50)
	lon = CHtoWGSlng(x + 50, y + 50)

	row_new.extend(r.create_features_withDist(lon, lat))

	return row_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--distance", help="Create also distance features.", action='store_true')
	parser.add_argument("-n", "--nWorkers", help="Number of parallel processes.")


	args = parser.parse_args()

	bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]
	data = []

	for x in range(bounds[0], bounds[1] + 1, 100):
		for y in range(bounds[2], bounds[3] + 1, 100):
			data.append([x, y])

	nWorkers = 1
	if args.nWorkers:
		nWorkers = args.nWorkers

	# r = Requestor.Requestor("zurich")

	print("Starting feature generation.")
	start_time = time.time()
	data_new = preproc_landuse_features_parallel(data, n_workers=nWorkers)
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = "mapdata"

	if args.distance:
		filenew = filenew + "_landUse_withDistances.csv"
	else:
		filenew = filenew + "_landUse.csv"

	with open(paths.lurdata + filenew, 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data_new:
			wr.writerow(row)

	print("Done! File saved as {}.".format(filenew))
