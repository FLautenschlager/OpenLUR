import math


def asRadians(degrees):
	return degrees * math.pi / 180


def asDegree(radians):
	return radians * 180 / math.pi


def coordToMeter(point):
	lon = point.geo_lon
	lat = point.geo_lat
	x = degree_equator * (lon - reference_geo['lon']) * reference_geo['cos_lat']
	y = degree_equator * (lat - reference_geo['lat'])
	return x, y


def meterToCoord(x, y):
	lon = x / (degree_equator * reference_geo['cos_lat']) + reference_geo['lon']
	lat = y / degree_equator + reference_geo['lat']
	return lon, lat


reference_geo = {'lat': 44.996474999999997,
                 'lon': 7.5438890000000001}  # the most southern and the most eastern coordinates

reference_geo['cos_lat'] = math.cos(asRadians(reference_geo['lat']))
earth_circumference = 40075160  # at equator
degree_equator = earth_circumference / 360
