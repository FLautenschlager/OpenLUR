import psycopg2
from operator import add


class Requestor:

    def __init__(self, database):
        self.conn = psycopg2.connect(dbname=database, user="docker", password="docker", port="5432", host="172.18.0.2")
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

        query = query[
                :-2] + "FROM planet_osm_polygon WHERE ST_DWithin(geo, geography(ST_MakePoint(%s, %s)), %s) AND {} = %s;".format(
            key)
        additional_values.append(lon_query)
        additional_values.append(lat_query)
        additional_values.append(max(radii))
        additional_values.append(value)
        # pre = time.time()
        self.cur.execute(query, tuple(additional_values))
        # print("Poly needed {}".format(time.time() - pre))
        return {"{}_{}m".format(value, d): (0 if v is None else v) for d, v in zip(radii, self.cur.fetchone())}

    def query_osm_line(self, lon_query, lat_query, radius, key, value):
        # pre = time.time()
        self.cur.execute(
            "SELECT sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s)))) FROM planet_osm_line WHERE ST_DWithin(geo, geography(ST_MakePoint(%s, %s)), %s) AND {} = %s;".format(
                key),
            (lon_query, lat_query, radius, lon_query, lat_query, radius, value))
        # print("Poly needed {}".format(time.time() - pre))
        return {"{}_{}m".format(value, radius): self.cur.fetchone()[0]}

    def query_osm_highway(self, lon_query, lat_query, radii):
        query = "SELECT "
        basic_query = "sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
        additional_values = []
        for radius in radii:
            query = query + basic_query + " , "
            additional_values.append(lon_query)
            additional_values.append(lat_query)
            additional_values.append(radius)

        query = query[:-2] + "FROM planet_osm_line WHERE ST_DWithin(geo, geography(ST_MakePoint(%s, %s)), %s)"
        additional_values.append(lon_query)
        additional_values.append(lat_query)
        additional_values.append(max(radii))

        qmotor = query + " AND highway = 'motorway';"
        qtrunk = query + " AND highway = 'trunk';"
        qprimary = query + " AND highway = 'primary';"
        qsecondary = query + " AND highway = 'secondary';"

        # pre = time.time()
        self.cur.execute(qmotor, tuple(additional_values))
        motor = [0 if v is None else v for v in self.cur.fetchone()]
        # print("Motorquery needed {}".format(time.time()-pre))

        # pre = time.time()
        self.cur.execute(qtrunk, tuple(additional_values))
        trunk = [0 if v is None else v for v in self.cur.fetchone()]
        # trunk = list(self.cur.fetchone())
        # print("TRUNK needed {}".format(time.time() - pre))

        # pre = time.time()
        self.cur.execute(qprimary, tuple(additional_values))
        primary = [0 if v is None else v for v in self.cur.fetchone()]
        # print("PRIM needed {}".format(time.time() - pre))

        # pre = time.time()
        self.cur.execute(qsecondary, tuple(additional_values))
        secondary = [0 if v is None else v for v in self.cur.fetchone()]
        # print("SEC needed {}".format(time.time() - pre))

        return {"bigRoad_{}m".format(d): (0 if v is None else v) for d, v in
                zip(radii, list(map(add, map(add, motor, trunk), map(add, primary, secondary))))}

    def query_osm_local_road(self, lon_query, lat_query, radii):
        query = "SELECT "
        basic_query = "sum(ST_Length(ST_Intersection(geo, ST_Buffer(geography(ST_SetSRID(ST_MakePoint(%s, %s),4326)), %s))))"
        additional_values = []
        for radius in radii:
            query = query + basic_query + " , "
            additional_values.append(lon_query)
            additional_values.append(lat_query)
            additional_values.append(radius)

        query = query[:-2] + "FROM planet_osm_line WHERE ST_DWithin(geo, geography(ST_MakePoint(%s, %s)), %s) "
        additional_values.append(lon_query)
        additional_values.append(lat_query)
        additional_values.append(max(radii))
        query_tert = query + " AND highway = 'tertiary';"
        query_res = query + " AND highway = 'residential';"

        # pre = time.time()
        self.cur.execute(query_tert, tuple(additional_values))
        tert = [0 if v is None else v for v in self.cur.fetchone()]
        # print("TERT needed {}".format(time.time() - pre))

        # pre = time.time()
        self.cur.execute(query_res, tuple(additional_values))
        res = [0 if v is None else v for v in self.cur.fetchone()]
        # print("RES needed {}".format(time.time() - pre))
        # print(query_res)

        return {"smallRoad_{}m".format(d): (0 if v is None else v) for d, v in zip(radii, list(map(add, res, tert)))}

    def query_osm_line_distance(self, lon_query, lat_query, key, value):
        query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_line WHERE {} = %s;".format(
            key)

        # pre = time.time()
        self.cur.execute(query, (lon_query, lat_query, value))
        # print("DistLine needed {}".format(time.time() - pre))
        return {value: self.cur.fetchone()[0]}

    def query_osm_point_distance(self, lon_query, lat_query, key, value):
        query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_point WHERE {} = %s;".format(
            key)

        # pre = time.time()
        self.cur.execute(query, (lon_query, lat_query, value))
        # print("DistPoint needed {}".format(time.time() - pre))
        return {value: self.cur.fetchone()[0]}

    def query_osm_polygon_distance(self, lon_query, lat_query, key, value):
        query = "SELECT min(ST_Distance(geo, geography(ST_SetSRID(ST_MakePoint(%s,%s),4326)))) FROM planet_osm_polygon WHERE {} = %s;".format(
            key)

        # pre = time.time()
        self.cur.execute(query, (lon_query, lat_query, value))
        # print("DistPoly needed {}".format(time.time() - pre))
        return {value: self.cur.fetchone()[0]}



    def create_features(self, lon, lat):
        features = {}
        try:
            features = {
                **(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "commercial")),
                **(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "industrial")),
                **(self.query_osm_polygone(lon, lat, list(range(50, 3050, 50)), "landuse", "residential")),
                **(self.query_osm_highway(lon, lat, list(range(50, 1550, 50)))),
                **(self.query_osm_local_road(lon, lat, list(range(50, 1550, 50)))),
                **(self.query_osm_point_distance(lon, lat, 'highway', 'traffic_signals')),
                **(self.query_osm_line_distance(lon, lat, 'highway', 'motorway')),
                **(self.query_osm_line_distance(lon, lat, 'highway', 'primary')),
                **(self.query_osm_polygon_distance(lon, lat, 'landuse', 'industrial'))
            }
        except Exception as e:
            print(e)
            print("error at point {}, {}".format(lat,lon))

            return {}
        return features

    def close(self):
        self.cur.close()
        self.conn.close()
