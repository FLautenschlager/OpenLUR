#!/bin/bash

# A script for loading .osm.pbf into postgis with creation
set -e

echo "creating database $1 from file $2"

createdb $1
echo "Database $1 created"

psql -d $1 -c 'CREATE EXTENSION postgis; CREATE EXTENSION hstore;'

osm2pgsql --create --database $1 -C 10000 $2
