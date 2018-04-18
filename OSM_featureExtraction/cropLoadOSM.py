#!/usr/bin/python3

from utils import paths
import sys
import subprocess
import argparse
import psycopg2
import time

def checkDBexists(dbname):
    try:
        conn = psycopg2.connect("dbname=template1")
    except:
        print('I am unable to connect to the database.')

    cur = conn.cursor()
    cur.execute("""select exists(SELECT datname FROM pg_catalog.pg_database WHERE lower(datname) = lower('{}'));""".format(dbname))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def crop(infile, outfile, latmin, latmax, lonmin, lonmax):

    start = time.time()
    print("Cropping OSM file")
    process = subprocess.call(["osmconvert", infile, "-b={},{},{},{}".format(lonmin, latmin, lonmax, latmax), "-o={}".format(outfile)])
    print("time needed: {} seconds.".format(time.time()-start))

def loadDB(infile, dbname):
    print("Drop and create DB")
    try:
        conn = psycopg2.connect("dbname=template1")
    except:
        print('I am unable to connect to the database.')

    cur = conn.cursor()
    conn.set_isolation_level(0)

    cur.execute("""DROP DATABASE IF EXISTS {};""".format(dbname))

    #cur.fetchall()
    cur.execute("""CREATE DATABASE {};""".format(dbname))
    #cur.fetchall()

    cur.close()
    conn.close()

    print("Create extensions")
    try:
        conn_new = psycopg2.connect("dbname={}".format(dbname))
        conn_new.autocommit = True
    except:
        print('I am unable to connect to the database {}.'.format(dbname))

    cur_new = conn_new.cursor()

    print("create extensions")
    cur_new.execute("CREATE EXTENSION postgis; CREATE EXTENSION hstore;")


    print("Load data to db")
    process = subprocess.call(["osm2pgsql", "--create", "--database", dbname, "-C", "10000", infile])

    print("Creating indexes")
    fd = open('OSM_featureExtraction/table_geography_creation.sql', 'r')
    sqlFile = fd.read()
    fd.close()

    sqlCommands = sqlFile.split('\n')
    for command in sqlCommands:
        if command:
            #print(command)
            try:
                cur_new.execute(command)
            except Exception as msg:
                print(msg)
                print("Command skipped: ", command)

    print("Created Tables:")
    table_name = ["planet_osm_polygon", "planet_osm_line", "planet_osm_point"]
    for table in table_name:
        cur_new.execute("SELECT COUNT(*) from {};".format(table))
        print(table + ": {} rows".format(cur_new.fetchone()))

    cur_new.close()
    conn_new.close()


def cropLoad(infile, dbname, latmin, latmax, lonmin, lonmax):
    outfile = paths.osmdir + '{}.osm.pbf'.format(dbname)
    crop(infile, outfile, latmin, latmax, lonmin, lonmax)
    loadDB(outfile, dbname)

def main(infile, dbname, latmin, latmax, lonmin, lonmax):
    cropLoad(paths.osmdir + infile, dbname, latmin, latmax, lonmin, lonmax)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='.osm.pbf file with OSM data')
    parser.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser.add_argument('latmin', type=float, help='minimum latitude')
    parser.add_argument('latmax', type=float, help='maximum latitude')
    parser.add_argument('lonmin', type=float, help='minimum longitude')
    parser.add_argument('lonmax', type=float, help='maximum longitude')

    args = parser.parse_args()
    starttime = time.time()
    main(args.infile, args.dbname.lower(), args.latmin, args.latmax, args.lonmin, args.lonmax)
    print("total time used: {} seconds".format(time.time()-starttime))
