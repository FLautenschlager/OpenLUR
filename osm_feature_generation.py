import argparse
import time

import pandas as pd

from OSM_featureExtraction.database_generation_utils import create_db
from OSM_featureExtraction.FeatureGenerator import FeatureGenerator


def create_prediction_features(arguments):
    dbname = arguments.dbname
    latmin = arguments.latmin
    latmax = arguments.latmax
    lonmin = arguments.lonmin
    lonmax = arguments.lonmax
    osmfile = arguments.osmfile
    n_workers = arguments.processors
    create_db(dbname, latmin, latmax, lonmin, lonmax, osmfile=osmfile, rebuild=arguments.rebuild)

    featuretime = time.time()
    fg = FeatureGenerator(dbname)
    fg.generateMap(latmin, latmax, lonmin, lonmax, granularity=0.001)
    fg.preproc_landuse_features_parallel(n_workers)
    featuretime = time.time() - featuretime

    print("Time needed for the feature extraction: {}s".format(featuretime))
    return fg.saveFeatures()


def create_file_features(arguments):
    df = pd.read_csv(arguments.file)
    latmin = df.latitude.min()
    latmax = df.latitude.max()
    lonmin = df.longitude.min()
    lonmax = df.longitude.max()
    create_db(arguments.dbname, latmin, latmax, lonmin, lonmax, osmfile=arguments.osmfile, rebuild=arguments.rebuild)

    print(df.head())
    featuretime = time.time()
    fg = FeatureGenerator(arguments.dbname, filename=arguments.file)
    fg.set_data_from_pandas(df, value=arguments.value)
    fg.preproc_landuse_features_parallel(arguments.processors)
    featuretime = time.time() - featuretime

    print("Time needed for the feature extraction: {}s".format(featuretime))
    return fg.saveFeatures()


def standardparsers(subparser):
    subparser.add_argument('-f', '--osmfile', type=str, help='path to .osm.pbf file if already present', default=None)
    subparser.add_argument('-p', '--processors', type=int, help='Number of workers to use for parallel processes',
                           default=1)
    subparser.add_argument('-r', '--rebuild', action='store_true', help='rebuild database')
    return subparser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_map = subparsers.add_parser("map")
    parser_map.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser_map.add_argument('latmin', type=float, help='minimum latitude')
    parser_map.add_argument('latmax', type=float, help='maximum latitude')
    parser_map.add_argument('lonmin', type=float, help='minimum longitude')
    parser_map.add_argument('lonmax', type=float, help='maximum longitude')

    parser_map = standardparsers(parser_map)
    parser_map.set_defaults(func=create_prediction_features)

    parser_file = subparsers.add_parser("file")
    parser_file.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser_file.add_argument("file", type=str, help="csv-file with lon, lat and value")
    parser_file.add_argument("-v", "--value", type=str, help="column of the value", default="value")
    parser_file = standardparsers(parser_file)
    parser_file.set_defaults(func=create_file_features)

    args = parser.parse_args()
    print(args)
    args.func(args)
