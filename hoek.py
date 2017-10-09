from os.path import expanduser
import scipy.io as sio
import math
import numpy as np
import psycopg2

import pyproj
latlong = pyproj.Proj(proj="latlon")
swiss = pyproj.Proj(init='EPSG:21781')
transformer = lambda x, y: pyproj.transform(swiss, latlong, x, y)
transformer_re = lambda x, y: pyproj.transform(latlong, swiss, x, y)

conn = psycopg2.connect(dbname="zurich")
cur = conn.cursor()

(8.543336, 47.398884, 1500, "landuse", "industrial")


def query_osm_polygone(x, y, radius, key, value):

	lon, lat = transformer_re(x,y)
	cur.execute(
		"SELECT sum(ST_Area(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_polygon WHERE {} = %s;".format(
			key),
		(lon, lat, radius, value))
	return cur.fetchone()[0]


def query_osm_line(x, y, radius, key, value):
	lon, lat = transformer_re(x, y)
	cur.execute(
		"SELECT sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE {} = %s;".format(
			key),
		(lon, lat, radius, value))
	return cur.fetchone()[0]


rootdir = expanduser("~/Data/OpenSense/Shared/UFP_Delivery_Lautenschlager/matlab/")
datadir = rootdir + "data/seasonal_maps/"
data = sio.loadmat(datadir + "filt/pm_01072012_31092012_filtered.mat")['data']

IND_AVG_DATA = 3
GEO_ACC = 4
print("Shape before cleaning: ", data.shape)
data = data[data[:, GEO_ACC] < 3, :]  # Remove inaccurate data
data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

bounds = sio.loadmat(rootdir + "bounds")['bounds']

print("Shape after cleaning: ", data.shape)

LAT = 1
LON = 2
pm_ha = []
for x in range(bounds[0, 0], bounds[0, 1], 100):
	for y in range(bounds[0, 2], bounds[0, 3], 100):

		# Fetch data in the bounding box
		temp = data[(data[:, LAT] >= x) & (data[:, LAT] < (x + 100)) & (data[:, LON] >= y) & (data[:, LON] < (y + 100)),
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

			try:
				indus = query_osm_polygone(x, y, 1000, "landuse", "industrial")
			except:
				print(x,y)
				print(transformer_re(x,y))

			pm_num.extend(indus)
			pm_ha.append(pm_num)

del data

LAT = 0
LON = 1

print("Training shape with tram depots: ({}, {})".format(len(pm_ha), len(pm_ha[1])))

# Tram depots
ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500], [681700, 249500], [683700, 251500], [679400, 248500],
             [683400, 249900], [683400, 249800], [682500, 243400]]

# check if Tram depot in bounding box
pm_ha_numpy = np.array(pm_ha)
for depot in ha_depots:
	pm_ha_numpy = pm_ha_numpy[~((pm_ha_numpy[:, LAT] == depot[LAT]) & (pm_ha_numpy[:, LON] == depot[LON])), :]

pm_ha = pm_ha_numpy.tolist()

del pm_ha_numpy
