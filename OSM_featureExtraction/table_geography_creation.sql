CREATE EXTENSION btree_gist;

ALTER TABLE planet_osm_roads ADD column geo geography;
UPDATE planet_osm_roads SET geo = geography(ST_Transform(way, 4326));
CREATE INDEX road_gix ON planet_osm_roads USING GIST (geo);
VACUUM ANALYZE planet_osm_roads;

ALTER TABLE planet_osm_point ADD column geo geography;
UPDATE planet_osm_point SET geo = geography(ST_Transform(way, 4326));
CREATE INDEX point_gix ON planet_osm_point USING GIST (geo);
VACUUM ANALYZE planet_osm_point;

ALTER TABLE planet_osm_line ADD column geo geography;
UPDATE planet_osm_line SET geo = geography(ST_Transform(way, 4326));
CREATE INDEX highway_line_simple ON planet_osm_line USING gist(geo, highway);
VACUUM ANALYZE planet_osm_line;

ALTER TABLE planet_osm_polygon ADD column geo geography;
UPDATE planet_osm_polygon SET geo = geography(ST_Transform(way, 4326));
CREATE INDEX landuse_poly_simple ON planet_osm_polygon USING gist(geo, landuse);
VACUUM ANALYZE planet_osm_polygon;
