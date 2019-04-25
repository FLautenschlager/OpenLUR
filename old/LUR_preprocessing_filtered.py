import argparse
import csv
import sys
import time

import scipy.io as sio

from OSM_featureExtraction import OSMRequestor
from experiments import paths
from utils.wgs84_ch1903 import *


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
	parser.add_argument("-a", "--all",
	                    help="If using -n or --fileNumber, this mutates to extracting features for all tiles, not only the selected 200.",
	                    action='store_true')

	args = parser.parse_args()

	r = OSMRequestor.Requestor("zurich")

	files = ["pm_ha_ext_01042012_30062012.mat",
	         "pm_ha_ext_01072012_31092012.mat",
	         "pm_ha_ext_01102012_31122012.mat",
	         "pm_ha_ext_01012013_31032013.mat"]

	files_all = ["pm_01042012_30062012_filtered_ha.csv",
	             "pm_01072012_31092012_filtered_ha.csv",
	             "pm_01102012_31122012_filtered_ha.csv",
	             "pm_01012013_31032013_filtered_ha.csv"]

	if args.file:
		file = args.file
	elif args.fileNumber:
		if args.all:
			file = files_all[args.fileNumber]
		else:
			file = files[args.fileNumber]
	else:
		print("Please specify file (-f/--file) or filenumber (-n/--fileNumber)")
		sys.exit(0)

	print("Loading file {}.".format(file))

	if args.all:
		data = sio.loadmat(paths.hadatadir + file)['pm_ha']
	else:
		data = sio.loadmat(paths.extdatadir + file)['pm_ha']

	print(data.shape)
	print("Starting feature generation.")
	start_time = time.time()
	data_new = preproc_landuse_features(data[:, 0:3])
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = file[:-4]

	if args.all:
		filenew = filenew + "_all"

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
