import csv

import psycopg2

from utils import paths
from utils.wgs84_ch1903 import *

conn = psycopg2.connect(dbname="zurich")
cur = conn.cursor()


# (8.543336, 47.398884, 1500, "landuse", "industrial")


def query_osm_polygone(lon_query, lat_query, radii, key, value):
	query = "SELECT "
	basic_query = "sum(ST_Area(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[:-2] +  "FROM planet_osm_polygon WHERE {} = %s;".format(key)
	additional_values.append(value)
	cur.execute(query,tuple(additional_values))
	return list(cur.fetchone())


def query_osm_line(lon_query, lat_query, radius, key, value):
	cur.execute(
		"SELECT sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE {} = %s;".format(
			key),
		(lon_query, lat_query, radius, value))
	return cur.fetchone()[0]


def query_osm_highway(lon_query, lat_query, radii):
	query = "SELECT "
	basic_query = "sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[:-2] +  "FROM planet_osm_line WHERE highway = 'motorway' OR highway = 'trunk' OR highway = 'primary' OR highway = 'secondary';"

	cur.execute(query,tuple(additional_values))
	return list(cur.fetchone())

def query_osm_local_road(lon_query, lat_query, radii):
	query = "SELECT "
	basic_query = "sum(ST_Length(ST_Intersection(geog, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[:-2] +  "FROM planet_osm_line WHERE highway = 'tertiary' OR highway = 'residential';"

	cur.execute(query,tuple(additional_values))
	return list(cur.fetchone())

def query_osm_line_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geog, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_line WHERE {} = %s;".format(key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()

def query_osm_point_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geog, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_point WHERE {} = %s;".format(key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()

def query_osm_polygon_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geog, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_polygon WHERE {} = %s;".format(key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()



def create_features(lon, lat):

	features = []
	try:
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "commercial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "industrial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "residential")))

		features.extend(query_osm_highway(lon, lat, list(range(50,1550,50))))
		features.extend(query_osm_local_road(lon, lat, list(range(50,1550,50))))

		features.extend(query_osm_point_distance(lon, lat, 'highway', 'traffic_signals'))
		features.extend(query_osm_line_distance(lon, lat, 'highway', 'motorway'))
		features.extend(query_osm_line_distance(lon, lat, 'highway', 'primary'))
		features.extend(query_osm_polygon_distance(lon, lat, 'landuse', 'industrial'))
	except Exception as e:
		print(e)
		print("error")
		print(lon, lat)
		features.extend([0 for _ in range(244)])

	return features


def create_features_from_SwissCoord(x, y):
	lat = CHtoWGSlat(x, y)
	lon = CHtoWGSlng(x, y)
	return create_features(lon, lat)



def preproc_landuse_features(data):

	data_new = []
	for row in data:
		x = row[0]
		y = row[1]
		m = row[2]
		row_new = [x, y, m]
		lat = CHtoWGSlat(x + 50, y + 50)
		lon = CHtoWGSlng(x + 50, y + 50)

		row_new.extend(create_features(lon, lat))

		data_new.append(row_new)

	return data_new


if __name__ == "__main__":

	file = "pm_01072013_31092013_filtered_ha.csv"
	data = []

	with open(paths.hadata + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	#data = preproc_land_use(paths.filtereddatadir + file)
	data_new = preproc_landuse_features(data)

	with open(paths.lurdata + file[:-4] + "_landUse.csv", 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data_new:
			wr.writerow(row)
