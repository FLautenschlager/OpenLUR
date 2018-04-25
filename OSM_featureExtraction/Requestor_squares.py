"""
Similar to Requestor.py but uses square buffers instead of regular round
buffers. This Requestor also expects Lv03 (Swiss) coordinates as it is easier to
deal with squares when the coordinates are meters.
"""
import math
import psycopg2
from operator import add
import time


# SELECT sum(ST_Area(ST_Intersection(geo, geography(ST_Transform(ST_SetSRID(ST_MakePolygon(ST_GeomFromText('LINESTRING(685500 248500, 685600 248500, 685600 248600, 685500 248600, 685500 248500)')),21781),4326))))) FROM planet_osm_polygon;
# SELECT sum(ST_Area(ST_Intersection(geo, geography(ST_Transform(ST_SetSRID(ST_MakePolygon(ST_GeomFromText('LINESTRING(685500 248500, 685600 248500, 685600 248600, 685500 248600, 685500 248500)')),21781),4326))))) FROM planet_osm_polygon WHERE ST_Touches(geo,  geography(ST_Transform(ST_SetSRID(ST_MakePolygon(ST_GeomFromText('LINESTRING(685500 248500, 685600 248500, 685600 248600, 685500 248600, 685500 248500)')),21781),4326)));

# SELECT sum(ST_Area(ST_Intersection(geo, geography(ST_Transform(ST_SetSRID(ST_MakeBox2D(ST_MakePoint(685500, 248500), ST_MakePoint(685600, 248600)),21781),4326))))) FROM planet_osm_polygon;
# SELECT sum(ST_Area(ST_Intersection(geo, geography(ST_Transform(ST_SetSRID(ST_MakeBox2D(ST_MakePoint(685500, 248500), ST_MakePoint(685600, 248600)),21781),4326))))) FROM planet_osm_polygon WHERE ST_Contains(geo, geography(ST_Transform(ST_SetSRID(ST_MakeBox2D(ST_MakePoint(685500, 248500), ST_MakePoint(685600, 248600)),21781),4326)));


# Auxiliary variables
SQUARE_GEOMETRY = "geography(ST_Transform(ST_SetSRID(ST_MakeBox2D(ST_MakePoint(%s, %s), ST_MakePoint(%s, %s)),21781),4326))"
POINT_GEOMETRY = "geography(ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s),21781),4326))"


class Requestor:

    def __init__(self):
        self.conn = psycopg2.connect(
            dbname='osm', user='postgres', password='themoreyouknow', host='localhost', port=5432)
        self.cur = self.conn.cursor()

    def query_osm_polygon(self, y_query, x_query, lengths, key, value):
        query = "SELECT "
        basic_query = "sum(ST_Area(ST_Intersection(geo, {})))".format(
            SQUARE_GEOMETRY)
        additional_values = []
        for length in lengths:
            query = query + basic_query + " , "
            additional_values.append(y_query - length / 2)
            additional_values.append(x_query - length / 2)
            additional_values.append(y_query + length / 2)
            additional_values.append(x_query + length / 2)

        query = query[:-2] + \
            """FROM planet_osm_polygon WHERE ST_DWithin(geo, {}, %s) AND "{}" = %s;""".format(
                POINT_GEOMETRY, key)
        additional_values.append(y_query)
        additional_values.append(x_query)
        # Distance to the corner of the square plus a meter for good measure
        additional_values.append(math.hypot(max(lengths)/2, max(lengths)/2) + 1)#math.sqrt(2 * (max(lengths) ** 2)) + 1)
        additional_values.append(value)
        #pre = time.time()
        self.cur.execute(query, tuple(additional_values))
        #print("Poly needed {}".format(time.time() - pre))
        return [0 if v is None else v for v in self.cur.fetchone()]

    def query_osm_highway(self, y_query, x_query, lengths):
        query = "SELECT "
        basic_query = "sum(ST_Length(ST_Intersection(geo, {})))".format(
            SQUARE_GEOMETRY)
        additional_values = []
        for length in lengths:
            query = query + basic_query + " , "
            additional_values.append(y_query - length / 2)
            additional_values.append(x_query - length / 2)
            additional_values.append(y_query + length / 2)
            additional_values.append(x_query + length / 2)

        query = query[:-2] + \
            "FROM planet_osm_line WHERE ST_DWithin(geo, {}, %s)".format(
                POINT_GEOMETRY)
        additional_values.append(y_query)
        additional_values.append(x_query)
        additional_values.append(math.hypot(max(lengths)/2, max(lengths)/2) + 1)#math.sqrt(2 * max(lengths) ** 2) + 1)

        queries = {
            "motor": query + " AND highway = 'motorway';",
            "trunk": query + " AND highway = 'trunk';",
            "primary": query + " AND highway = 'primary';",
            "secondary": query + " AND highway = 'secondary';",
            "tertiary": query + " AND highway = 'tertiary';",
            # Note: Unclassified does not mean that it is not classified
            # This is a special road type smaller than tertiary (e.g. link villages)
            "unclassified": query + " AND highway = 'unclassified';",
            "residential": query + " AND highway = 'residential';"
        }

        results = {}
        for road_type, q in queries.items():
            #pre = time.time()
            self.cur.execute(q, tuple(additional_values))
            results[road_type] = [
                0 if v is None else v for v in self.cur.fetchone()]
            #print("{} needed {}".format(road_type, time.time()-pre))

        return results



    def query_osm_largest_road_type(self, y_query, x_query, lengths):
        road_lengths = self.query_osm_highway(y_query, x_query, lengths)

        road_type_hierarchy = ["motor", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]

        # Return 7 when no road exists
        results = [7] * len(lengths)

        for l_ind in range(len(lengths)):
            for rt_ind, rt in enumerate(road_type_hierarchy):
                # Return the index of the largest road type that exists
                # Note: larger roads have lower indices (0 = largest road type)
                if (road_lengths[rt][l_ind] > 0.0):
                    results[l_ind] = rt_ind
                    # Stop for this length since we found the largest road
                    break

        return results


    def query_osm_line_distance(self, y_query, x_query, key, value):
        # I just look for distance to lines closer than 9 km to speed queries up
        # dramatically. In zurich each roadtype is at most 8223 m away from each
        # point so it's fine for my use case.
        query = "SELECT min(ST_Distance(geo, {})) FROM planet_osm_line WHERE ST_DWITHIN(geo, {}, 9000) AND {} = %s;".format(POINT_GEOMETRY, POINT_GEOMETRY,
                                                                                              key)

        #pre = time.time()
        self.cur.execute(query, (y_query, x_query, y_query, x_query, value))
        #print("DistLine needed {}".format(time.time() - pre))
        return self.cur.fetchone()

    def query_osm_large_road_distance(self, y_query, x_query):
        query = "SELECT min(ST_Distance(geo, {})) FROM planet_osm_line WHERE highway = 'motorway' OR highway = 'trunk' OR highway = 'primary' OR highway = 'secondary';".format(POINT_GEOMETRY)

        #pre = time.time()
        self.cur.execute(query, (y_query, x_query))
        #print("DistLine needed {}".format(time.time() - pre))
        return self.cur.fetchone()

    def query_osm_elevation(self, y_query, x_query, max_ele=900, min_ele=200):
        """Find closest elevation point. Note: results in crap in zurich"""
        # query = "SELECT ele FROM planet_osm_point WHERE ele IS NOT NULL AND min(ST_Distance(geo, {}));".format(POINT_GEOMETRY)
        query = "SELECT min(ST_Distance(geo, {})), ele FROM planet_osm_point WHERE ele IS NOT NULL GROUP BY ele;".format(
            POINT_GEOMETRY)

        #pre = time.time()
        self.cur.execute(query, (y_query, x_query))
        #print("Elevation needed {}".format(time.time() - pre))

        result = None
        while(result is None):
            result = self.cur.fetchone()[1]
            print(result)
            if(isinstance(result, str)):
                result = result.split('m')[0]

            try:
                result = float(result)
            except Exception as e:
                print(e)
                result = None
                continue

            if result > max_ele or result < min_ele:
                result = None

        return result

    def create_features(self, y, x):
        features = []
        try:
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "landuse", "commercial")))
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "landuse", "industrial")))
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "landuse", "residential")))
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "leisure", "park")))
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "landuse", "grass")))
            features.extend(list(self.query_osm_polygon(
                y, x, [100, 200, 300], "natural", "water")))

            features.extend(self.query_osm_largest_road_type(
                y, x, [100, 200, 300]))
            
            # features.extend(list(self.query_osm_large_road_distance(y, x)))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'motorway')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'trunk')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'primary')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'secondary')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'tertiary')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'unclassified')))
            features.extend(list(self.query_osm_line_distance(y, x, 'highway', 'residential')))

        except Exception as e:
            print(e)
            print("error")
            print(y, x)
            features.extend([0 for _ in range(28)])

        return features

    def close(self):
        self.cur.close()
        self.conn.close()
