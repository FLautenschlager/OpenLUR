import csv

import argparse
import scipy.io as sio
import psycopg2
import time

import paths
from wgs84_ch1903 import *
import Requestor

def create_features_from_SwissCoord(x, y):
	lat = CHtoWGSlat(x, y)
	lon = CHtoWGSlng(x, y)
	return r.create_features(lon, lat)


def preproc_landuse_features(data):
	if args.distance:
		func = r.create_features_withDist
	else:
		func = r.create_features

	data_new = []
	for row in data:
		x = row[0]
		y = row[1]
		m = row[2]
		row_new = [x, y, m]
		lat = CHtoWGSlat(x + 50, y + 50)
		lon = CHtoWGSlng(x + 50, y + 50)

		row_new.extend(func(lon, lat))

		data_new.append(row_new)

	return data_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help="Define an input file.")
	parser.add_argument("-n", "--fileNumber", type=int, help="Input file as number of season.")
	parser.add_argument("-d", "--distance", help="Create also distance features.", action='store_true')

	args = parser.parse_args()

	r = Requestor.Requestor("zurich")

	files = ["pm_ha_ext_01042012_30062012.mat",
	         "pm_ha_ext_01072012_31092012.mat",
	         "pm_ha_ext_01102012_31122012.mat",
	         "pm_ha_ext_01012013_31032013.mat"]

	if args.file:
		file = args.file
	elif args.fileNumber:
		file = files[args.fileNumber]
	else:
		# file = "pm_ha_ext_01012013_31032013.mat"
		file = "pm_ha_ext_01042012_30062012.mat"
	# file = "pm_ha_ext_01072012_31092012.mat"
	# file = "pm_ha_ext_01102012_31122012.mat"

	print("Loading file {}.".format(file))

	data = sio.loadmat(paths.extdatadir + file)['pm_ha']
	print(data.shape)
	print("Starting feature generation.")
	start_time = time.time()
	data_new = preproc_landuse_features(data[:, 0:3])
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = file[:-4]

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
