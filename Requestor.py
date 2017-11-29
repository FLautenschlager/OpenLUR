import psycopg2


class Requestor:

	def __init__(self, database):
		self.conn = psycopg2.connect(dbname=database)
		self.cur = self.conn.cursor()

	def query_osm_polygone(self, lon_query, lat_query, radii, key, value):
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
		self.cur.execute(query, tuple(additional_values))
		return list(self.cur.fetchone())

	def query_osm_line(self, lon_query, lat_query, radius, key, value):
		self.cur.execute(
			"SELECT sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE {} = %s;".format(
				key),
			(lon_query, lat_query, radius, value))
		return self.cur.fetchone()[0]

	def query_osm_highway(self, lon_query, lat_query, radii):
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

		self.cur.execute(query, tuple(additional_values))
		return list(self.cur.fetchone())

	def query_osm_local_road(self, lon_query, lat_query, radii):
		query = "SELECT "
		basic_query = "sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
		additional_values = []
		for radius in radii:
			query = query + basic_query + " , "
			additional_values.append(lon_query)
			additional_values.append(lat_query)
			additional_values.append(radius)

		query = query[:-2] + "FROM planet_osm_line WHERE highway = 'tertiary' OR highway = 'residential';"

		self.cur.execute(query, tuple(additional_values))
		return list(self.cur.fetchone())

	def query_osm_line_distance(self, lon_query, lat_query, key, value):
		query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_line WHERE {} = %s;".format(
			key)

		self.cur.execute(query, (lon_query, lat_query, value))
		return self.cur.fetchone()

	def query_osm_point_distance(self, lon_query, lat_query, key, value):
		query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_point WHERE {} = %s;".format(
			key)

		self.cur.execute(query, (lon_query, lat_query, value))
		return self.cur.fetchone()

	def query_osm_polygon_distance(self, lon_query, lat_query, key, value):
		query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_polygon WHERE {} = %s;".format(
			key)

		self.cur.execute(query, (lon_query, lat_query, value))
		return self.cur.fetchone()

	def create_features(self, lon, lat):
		features = []
		try:
			features.extend(list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial")))
			features.extend(list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial")))
			features.extend(
				list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "residential")))

			features.extend(self.query_osm_highway(lon, lat, list(range(50, 1550, 50))))
			features.extend(self.query_osm_local_road(lon, lat, list(range(50, 1550, 50))))

		except Exception as e:
			print(e)
			print("error")
			print(lon, lat)
			features.extend([0 for _ in range(240)])

		return features

	def create_features_withDist(self, lon, lat):
		features = []
		try:
			features.extend(list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial")))
			features.extend(list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial")))
			features.extend(
				list(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "residential")))

			features.extend(self.query_osm_highway(lon, lat, list(range(50, 1550, 50))))
			features.extend(self.query_osm_local_road(lon, lat, list(range(50, 1550, 50))))

			features.extend(self.query_osm_point_distance(lon, lat, 'highway', 'traffic_signals'))
			features.extend(self.query_osm_line_distance(lon, lat, 'highway', 'motorway'))
			features.extend(self.query_osm_line_distance(lon, lat, 'highway', 'primary'))
			features.extend(self.query_osm_polygon_distance(lon, lat, 'landuse', 'industrial'))
		except Exception as e:
			print(e)
			print("error")
			print(lon, lat)
			features.extend([0 for _ in range(244)])

		return features

	def close(self):
		self.cur.close()
		self.conn.close()
