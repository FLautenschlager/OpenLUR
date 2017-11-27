import csv
import scipy.io as sio
import psycopg2

import paths
from wgs84_ch1903 import *

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



def create_features(lon, lat):

	features = []
	try:
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "commercial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "industrial")))
		features.extend(list(query_osm_polygone(lon, lat, list(range(50,3050,50)), "landuse", "residential")))

		features.extend(query_osm_highway(lon, lat, list(range(50,1550,50))))
		features.extend(query_osm_local_road(lon, lat, list(range(50,1550,50))))
	except Exception as e:
		print(e)
		print("error")
		print(lon, lat)
		features.extend([0 for _ in range(240)])

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

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help="Define an input file.")
	parser.add_argument("-n", "--fileNumber", type=int, help="Input file as number of season.")

	args = parser.parse_args()

	files = ["pm_ha_ext_01042012_30062012.mat", "pm_ha_ext_01072012_31092012.mat", "pm_ha_ext_01102012_31122012.mat", "pm_ha_ext_01012013_31032013.mat"]

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
	data_new = preproc_landuse_features(data[:,0:3])
	print("Features generated in {} minutes!".format((time.time()-start_time)/60))

	filenew = file[:-4] + "_landUse_withDistances.csv"

	with open(paths.lurdata + filenew, 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data_new:
			wr.writerow(row)

	print("Done! File saved as {}.".format(filenew))
