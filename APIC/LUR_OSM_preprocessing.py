import csv
import argparse
import time

import paths
from APIC.local_coordinates import *
import Requestor


def create_features_from_localCoord(x, y):
	lon, lat = meterToCoord(x + 50, y + 50)
	return r.create_features(lon, lat)


def preproc_landuse_features(data):
	data_new = []

	if args.distance:
		func = r.create_features_withDist
	else:
		func = r.create_features

	for row in data:
		x = row[0]
		y = row[1]
		lon = row[2]
		lat = row[3]
		m = row[4]
		row_new = [x, y, m]

		row_new.extend(func(lat, lon))

		data_new.append(row_new)

	return data_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--distance", help="Create also distance features.", action='store_true')

	args = parser.parse_args()

	r = Requestor.Requestor("turin")

	data = []
	file = "turin_tiles_200.csv"
	with open(paths.apicdir + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	print(len(data))
	print("Starting feature generation.")
	start_time = time.time()
	data_new = preproc_landuse_features(data)
	print("Features generated in {} minutes!".format((time.time() - start_time) / 60))

	filenew = file[:-4]

	if args.distance:
		filenew = filenew + "_landUse_withDistances.csv"
	else:
		filenew = filenew + "_landUse.csv"

	with open(paths.apicdir + filenew, 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data_new:
			wr.writerow(row)

	print("Done! File saved as {}.".format(filenew))
