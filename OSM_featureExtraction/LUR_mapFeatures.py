import argparse
import csv
import time
import numpy as np

import scipy.io as sio
from joblib import Parallel, delayed

from OSM_featureExtraction import Requestor
from utils import paths
from utils.wgs84_ch1903 import *


#
# def create_features_from_SwissCoord(x, y):
# 	lat = CHtoWGSlat(x, y)
# 	lon = CHtoWGSlng(x, y)
# 	return r.create_features(lon, lat)


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


def preproc_landuse_features_parallel(data, city, latlon=True, n_workers=1):
	if args.distance:
		func = preproc_single_with_dist
	else:
		func = preproc_single

	data_new = Parallel(n_jobs=n_workers, verbose=10)(delayed(func)((row, latlon, city)) for row in data)
	return data_new


def preproc_single(data):
	row, latlon, city = data
	r = Requestor.Requestor(city)
	if not latlon:
		x = row[0]
		y = row[1]
		row_new = [x, y, 0]
		lat = CHtoWGSlat(x + 50, y + 50)
		lon = CHtoWGSlng(x + 50, y + 50)
	else:
		lat = row[0]
		lon = row[1]
		row_new = [lat, lon, 0]

	row_new.extend(r.create_features(lon, lat))

	return row_new


def preproc_single_with_dist(data):
	row, latlon, city = data
	r = Requestor.Requestor(city)
	if not latlon:
		x = row[0]
		y = row[1]
		row_new = [x, y, 0]
		lat = CHtoWGSlat(x + 50, y + 50)
		lon = CHtoWGSlng(x + 50, y + 50)
	else:
		lat = row[0]
		lon = row[1]
		row_new = [lat, lon, 0]
	row_new.extend(r.create_features_withDist(lon, lat))

	return row_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--distance", help="Create also distance features.", action='store_true')
	parser.add_argument("-n", "--nWorkers", help="Number of parallel processes.")
	parser.add_argument("-r", "--region", help="chose region: Torino, Antwerp, London, Kassel, Zurich", type=str)

	args = parser.parse_args()

	latlon = True
	if args.region == "Torino":
		region = "italy"
		bounds = {'latmin': 44.9541, 'latmax': 45.1506, 'lonmin': 7.5057, 'lonmax': 7.7996}
	elif args.region == "Zurich":
		region = "switzerland"
		latlon = False
	elif args.region == "London":
		region = "greatbritain"
		bounds = {'latmin': 51.2744, 'latmax': 51.7066, 'lonmin': -0.5404, 'lonmax': 0.3323}
	elif args.region == "Antwerp":
		region = "belgium"
		bounds = {'latmin': 51.1346, 'latmax': 51.3225, 'lonmin': 4.1892, 'lonmax': 4.5834}
	elif args.region == "Kassel":
		region = "germany"
		bounds = {'latmin': 51.2355, 'latmax': 51.3932, 'lonmin': 9.3617, 'lonmax': 9.6460}
	elif args.region == "NewYork":
		region = "newyork"
		bounds = {'latmin': 40.5389, 'latmax': 40.8694, 'lonmin': -74.1378, 'lonmax': -73.7402}
	else:
		"Chose one of the given regions"
		region = "Null"

	data = []
	if region == "switzerland":
		bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]

		for x in range(bounds[0], bounds[1] + 1, 100):
			for y in range(bounds[2], bounds[3] + 1, 100):
				data.append([x, y])
	elif region == "Null":
		pass
	else:

		for lat in np.arange(bounds['latmin'], bounds['latmax'], 0.001):
			for lon in np.arange(bounds['lonmin'], bounds['lonmax'], 0.001):
				data.append([lat, lon])

	nWorkers = 1
	if args.nWorkers:
		nWorkers = int(args.nWorkers)

	# r = Requestor.Requestor("zurich")

	print("Starting feature generation. Total count of cells: {}".format(len(data)))
	start_time = time.time()
	data_new = preproc_landuse_features_parallel(data, region, latlon=latlon, n_workers=nWorkers)
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = "mapdata_{}".format(args.region)

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
