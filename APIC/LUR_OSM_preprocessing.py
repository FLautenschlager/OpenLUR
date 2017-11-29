import sys
from os.path import expanduser

import csv
import argparse
import numpy as np
import psycopg2
import time

homedir = expanduser("~/")
if (homedir + "Code/code-2017-land-use") not in sys.path:
	print("Adding path to sys.path: " + homedir + "code-2017-land-use")
	sys.path.append(homedir + "Code/code-2017-land-use")

import paths
from APIC.local_coordinates import *

conn = psycopg2.connect(dbname="turin")
cur = conn.cursor()


# (8.543336, 47.398884, 1500, "landuse", "industrial")


def query_osm_polygone(lon_query, lat_query, radii, key, value):
	query = "SELECT "
	basic_query = "sum(ST_Area(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[:-2] + "FROM planet_osm_polygon WHERE {} = %s;".format(key)
	additional_values.append(value)
	cur.execute(query, tuple(additional_values))
	return list(cur.fetchone())


def query_osm_line(lon_query, lat_query, radius, key, value):
	cur.execute(
		"SELECT sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE {} = %s;".format(
			key),
		(lon_query, lat_query, radius, value))
	return cur.fetchone()[0]


def query_osm_highway(lon_query, lat_query, radii):
	query = "SELECT "
	basic_query = "sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[
	        :-2] + "FROM planet_osm_line WHERE highway = 'motorway' OR highway = 'trunk' OR highway = 'primary' OR highway = 'secondary';"

	cur.execute(query, tuple(additional_values))
	return list(cur.fetchone())


def query_osm_local_road(lon_query, lat_query, radii):
	query = "SELECT "
	basic_query = "sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
	additional_values = []
	for radius in radii:
		query = query + basic_query + " , "
		additional_values.append(lon_query)
		additional_values.append(lat_query)
		additional_values.append(radius)

	query = query[:-2] + "FROM planet_osm_line WHERE highway = 'tertiary' OR highway = 'residential';"

	cur.execute(query, tuple(additional_values))
	return list(cur.fetchone())


def query_osm_line_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_line WHERE {} = %s;".format(
		key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()


def query_osm_point_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_point WHERE {} = %s;".format(
		key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()


def query_osm_polygon_distance(lon_query, lat_query, key, value):
	query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_polygon WHERE {} = %s;".format(
		key)

	cur.execute(query, (lon_query, lat_query, value))
	return cur.fetchone()


def create_features(lon, lat):
	features = []
	try:
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "residential")))

		features.extend(query_osm_highway(lon, lat, list(range(50, 1550, 50))))
		features.extend(query_osm_local_road(lon, lat, list(range(50, 1550, 50))))

	except Exception as e:
		print(e)
		print("error")
		print(lon, lat)
		features.extend([0 for _ in range(240)])

	return features


def create_features_withDist(lon, lat):
	features = []
	try:
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "residential")))

		features.extend(query_osm_highway(lon, lat, list(range(50, 1550, 50))))
		features.extend(query_osm_local_road(lon, lat, list(range(50, 1550, 50))))

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


def create_features_from_localCoord(x, y):
	lon, lat = meterToCoord(x + 50, y + 50)
	return create_features(lon, lat)


def preproc_landuse_features(data):
	data_new = []

	if args.distance:
		func = create_features_withDist
	else:
		func = create_features

	for row in data:
		x = row[0]
		y = row[1]
		lon = row[2]
		lat = row[3]
		m = row[4]
		row_new = [x, y, m]

		row_new.extend(func(lon, lat))

		data_new.append(row_new)

	return data_new


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--distance", help="Create also distance features.", action='store_true')

	args = parser.parse_args()

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
