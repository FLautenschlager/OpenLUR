#!/usr/bin/python3

from utils import paths
import sys
import subprocess
import argparse
import psycopg2

def crop(infile, outfile, latmin, latmax, lonmin, lonmax):
    
    print("Cropping OSM file")
    process = subprocess.Popen(["osmconvert", infile, "-b={},{},{},{}".format(lonmin, latmin, lonmax, latmax), "-o={}".format(outfile)], stdout=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()


def loadDB(infile, dbname):
    print("Drop and create DB")
    try:
        conn = psycopg2.connect("user='postgres'")
    except:
        print('I am unable to connect to the database.')

    cur = conn.cursor()
    conn.set_isolation_level(0)
    try:
        cur.execute("""DROP DATABASE IF EXISTS (%s);""", dbname)
    except:
        print('I cannot drop the database!')
    cur.fetchall()
    cur.execute("""CREATE DATABASE (%s);""", dbname)
    cur.fetchall()

    cur.close()
    conn.close()

    print("Create extensions")
    try:
        conn = psycopg2.connect("dbname='{}' user='postgres'".format(dbname))
    except:
        print('I am unable to connect to the database {}.'.format(dbname))

    cur = conn.cursor()
 
    cur.execute("""CREATE EXTENSION postgis; CREATE EXTENSION hstore;""")
    cur.fetchall()

    print("Load data to db")
    process = subprocess.Popen(["osm2pgsql", "--create", "--database", dbname, "-C", "10000", infile], stdout=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()

    print("Creating indexes")
    fd = open('table_geography_creation', 'r')
    sqlFile = fd.read()
    fd.close()

    sqlCommands = sqlFile.split(';')
    for command in sqlCommands:
        try: 
            cur.execute(command)
        except OperationalError as msg:
            print("Command skipped: ", msg)

    cur.close()
    conn.close()


def loadCrop(infile, dbname, latmin, latmax, lonmin, lonmax):
    outfile = paths.osmdir + '{}.osm.pbf'.format(dbname)
    crop(infile, outfile, latmin, latmax, lonmin, lonmax)
    loadDB(outfile, dbname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='.osm.pbf file with OSM data')
    parser.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser.add_argument('latmin', type=float, help='minimum latitude')
    parser.add_argument('latmax', type=float, help='maximum latitude')
    parser.add_argument('lonmin', type=float, help='minimum longitude')
    parser.add_argument('lonmax', type=float, help='maximum longitude')

    args = parser.parse_args()
    
    
    loadCrop(paths.osmdir + args.infile, args.dbname, args.latmin, args.latmax, args.lonmin, args.lonmax)


if __name__=='__main__':
    main()
