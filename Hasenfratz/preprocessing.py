import csv
import math

import numpy as np
import scipy.io as sio

from utils import paths


def clean_tramDepots(data_tram):
	LAT = 0
	LON = 1

	print("Training shape with tram depots: ({}, {})".format(len(data_tram), len(data_tram[1])))

	# Tram depots
	ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500], [681700, 249500], [683700, 251500],
	             [679400, 248500],
	             [683400, 249900], [683400, 249800], [682500, 243400]]

	# check if Tram depot in bounding box
	data_numpy = np.array(data_tram)
	for depot in ha_depots:
		data_numpy = data_numpy[~((data_numpy[:, LAT] == depot[LAT]) & (data_numpy[:, LON] == depot[LON])), :]

	data_tram = data_numpy.tolist()
	del data_numpy
	print("Training shape without tram depots: ({}, {})".format(len(data_tram), len(data_tram[1])))

	return data_tram


def preproc(data_dict_preproc):
	IND_AVG_DATA = 3
	GEO_ACC = 4
	print("Shape before cleaning: ", data_dict_preproc.shape)
	data_dict_preproc = data_dict_preproc[data_dict_preproc[:, GEO_ACC] < 3, :]  # Remove inaccurate data
	data_dict_preproc = data_dict_preproc[data_dict_preproc[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
	data_dict_preproc = data_dict_preproc[data_dict_preproc[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

	bounds = sio.loadmat(paths.rootdir + "bounds")['bounds']

	print("Shape after cleaning: ", data_dict_preproc.shape)

	LAT = 1
	LON = 2
	pm_ha = []
	for x in range(bounds[0, 0], bounds[0, 1], 100):
		for y in range(bounds[0, 2], bounds[0, 3], 100):

			# Fetch data in the bounding box
			temp = data_dict_preproc[
			       (data_dict_preproc[:, LAT] >= x) & (data_dict_preproc[:, LAT] < (x + 100)) & (
				       data_dict_preproc[:, LON] >= y) & (data_dict_preproc[:, LON] < (y + 100)), :]
			if temp.shape[0] != 0:
				# Calculate Statistics and dependent variable
				m = np.mean(temp[:, IND_AVG_DATA])

				pm_num = [x, y, m]

				pm_ha.append(pm_num)

	del data_dict_preproc

	return clean_tramDepots(pm_ha)


if __name__ == "__main__":
	file = "pm_01072013_31092013_filtered.mat"

	data = sio.loadmat(paths.filtereddatadir + file)['data']

	data = preproc(data)
	data = clean_tramDepots(data)

	with open(paths.hadata + file[:-4] + "_ha.csv", 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data:
			wr.writerow(row)
