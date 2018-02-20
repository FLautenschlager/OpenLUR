ALTER TABLE planet_osm_roads ADD column geo geography;
UPDATE planet_osm_roads SET geo = geography(ST_Transform(way, 4326));

ALTER TABLE planet_osm_point ADD column geo geography;
UPDATE planet_osm_point SET geo = geography(ST_Transform(way, 4326));

ALTER TABLE planet_osm_line ADD column geo geography;
UPDATE planet_osm_line SET geo = geography(ST_Transform(way, 4326));

ALTER TABLE planet_osm_polygon ADD column geo geography;
UPDATE planet_osm_polygon SET geo = geography(ST_Transform(way, 4326));

