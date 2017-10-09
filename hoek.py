from os.path import expanduser
import scipy.io as sio
import math
import numpy as np
import psycopg2
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from wgs84_ch1903 import *

conn = psycopg2.connect(dbname="zurich")
cur = conn.cursor()


# (8.543336, 47.398884, 1500, "landuse", "industrial")


def query_osm_polygone(lon_query, lat_query, radius, key, value):
	cur.execute(
		"SELECT sum(ST_Area(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_polygon WHERE {} = %s;".format(
			key),
		(lon_query, lat_query, radius, value))
	return cur.fetchone()[0]


def query_osm_line(lon_query, lat_query, radius, key, value):
	cur.execute(
		"SELECT sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE {} = %s;".format(
			key),
		(lon_query, lat_query, radius, value))
	return cur.fetchone()[0]


def query_osm_highway(lon_query, lat_query, radius):
	cur.execute(
		"SELECT sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%S, %S),4326)), %S)))) FROM planet_osm_line WHERE highway != NULL;",
		(lon_query, lat_query, radius))
	return cur.fetchone()[0]


def preproc_land_use(filename):
	data = sio.loadmat(filename)['data']

	IND_AVG_DATA = 3
	GEO_ACC = 4
	print("Shape before cleaning: ", data.shape)
	data = data[data[:, GEO_ACC] < 3, :]  # Remove inaccurate data
	data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
	data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

	bounds = sio.loadmat(rootdir + "Shared/UFP_Delivery_Lautenschlager/matlab/bounds")['bounds']

	print("Shape after cleaning: ", data.shape)

	LAT = 1
	LON = 2
	pm_ha = []
	for x in range(bounds[0, 0], bounds[0, 1], 100):
		for y in range(bounds[0, 2], bounds[0, 3], 100):

			# Fetch data in the bounding box
			temp = data[
			       (data[:, LAT] >= x) & (data[:, LAT] < (x + 100)) & (data[:, LON] >= y) & (data[:, LON] < (y + 100)),
			       :]
			if temp.shape[0] != 0:
				# Calculate Statistics and dependent variable
				m = np.mean(temp[:, IND_AVG_DATA])
				s = np.std(temp[:, IND_AVG_DATA])
				med = np.median(temp[:, IND_AVG_DATA])

				log = np.log(temp[:, IND_AVG_DATA])
				# log[log == -float('Inf')] = 0
				log = log[log != -float('Inf')]

				m_log = np.mean(log)
				s_log = np.std(log)

				pm_num = [x, y, m, temp.shape[0], s, m_log, s_log, med]

				lat = CHtoWGSlat(y + 50, x + 50)
				lon = CHtoWGSlng(y + 50, x + 50)

				industry = []
				highway = []
				for i in range(50, 1500, 50):
					try:
						industry.append(query_osm_polygone(lon, lat, i, "landuse", "industrial"))
						highway.append(query_osm_highway(lon, lat, i))
					except:
						print(x, y)
						print(lon, lat)
						industry.append(0)
						highway.append(0)

				pm_num.extend(industry)
				pm_num.extend(highway)
				pm_ha.append(pm_num)

	del data

	LAT = 0
	LON = 1

	print("Training shape with tram depots: ({}, {})".format(len(pm_ha), len(pm_ha[1])))

	# Tram depots
	ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500], [681700, 249500], [683700, 251500],
	             [679400, 248500],
	             [683400, 249900], [683400, 249800], [682500, 243400]]

	# check if Tram depot in bounding box
	pm_ha_numpy = np.array(pm_ha)
	for depot in ha_depots:
		pm_ha_numpy = pm_ha_numpy[~((pm_ha_numpy[:, LAT] == depot[LAT]) & (pm_ha_numpy[:, LON] == depot[LON])), :]

	pm_ha = pm_ha_numpy.tolist()
	print("Maximum value {}".format(pm_ha_numpy[:, 9:].max()))

	del pm_ha_numpy

	return pm_ha


def classification(train_data):
	model = LinearRegression()
	train_data_np = np.array(train_data)
	scores = cross_val_score(model, train_data_np[:, 8:], train_data_np[:, 2], cv=5)
	print(scores)


if __name__ == "__main__":
	rootdir = expanduser("~/Data/OpenSense/")
	datadir = rootdir + "Shared/UFP_Delivery_Lautenschlager/matlab/data/seasonal_maps/filt/"
	file = "pm_01072012_31092012_filtered.mat"

	pm_ha = preproc_land_use(datadir + file)

	with open(rootdir + file[:-4] + ".csv", 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in pm_ha:
			wr.writerow(row)

	classification(pm_ha)
