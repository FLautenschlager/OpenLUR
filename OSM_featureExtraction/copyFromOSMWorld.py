#!/usr/bin/python3

from utils import paths
import sys
import subprocess
import argparse
import psycopg2
import time
"""

Copy a *part* of a database to another one. See
<http://stackoverflow.com/questions/414849/whats-the-best-way-to-copy-a-subset-of-a-tables-rows-from-one-database-to-anoth>

With PostgreSQL, the only pure-SQL solution is to use COPY, which is
not available to the ordinary user.

Stephane Bortzmeyer <bortzmeyer@nic.fr>

"""
    def copy(dbname, latmin, latmax, lonmin, lonmax):
        table_name = ["planet_osm_polygon", "planet_osm_line", "planet_osm_point"]
        # List here the columns you want to copy. Yes, "*" would be simpler
        # but also more brittle.
        names = ["id", "uuid", "date", "domain", "broken", "spf"]
        constraint = "ST_Intersects(way, ST_GeomFromText('POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'))".format(lonmin, latmin, lonmin, latmax, lonmax, latmax, lonmax, latmin, lonmin, latmin)


        old_db = psycopg2.connect("dbname=osm_world")
        new_db = psycopg2.connect("dbname={}".format(dbname))
        old_cursor = old_db.cursor()
        old_cursor.execute("""SET TRANSACTION READ ONLY""") # Security
        new_cursor = new_db.cursor()
        old_cursor.execute("""SELECT * FROM {} WHERE {} """.format(table_name, constraint))
        print("{} rows retrieved for table {}.".format(old_cursor.rowcount, table_name))
        names = [desc[0] for desc in old_cursor.description]
        new_cursor.execute("""BEGIN""")
        placeholders = []
        namesandvalues = {}
        for name in names:
            placeholders.append("%%({})s".format(name))
        command = "INSERT INTO {} ({}) VALUES ({})".format(table_name, ",".join(names), ",".join(placeholders))
        for row in old_cursor.fetchall():
            i = 0
            for name in names:
                namesandvalues[name] = row[i]
                i = i + 1

            new_cursor.execute(command, namesandvalues)
        new_cursor.execute("""COMMIT""")
        old_cursor.close()
        new_cursor.close()
        old_db.close()
        new_db.close()

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
