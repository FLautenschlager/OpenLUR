import argparse
import time
from experiments import paths
import pandas as pd

from OSM_featureExtraction import database_generation_utils, download_planet
from OSM_featureExtraction.FeatureGenerator import FeatureGenerator


def checkDBexists(dbname):
    return False

def createDB(dbname, latmin, latmax, lonmin, lonmax, osmfile=None, nWorkers=1, rebuild=False):
    if not rebuild:
        if not checkDBexists(dbname):
            args.rebuild = True

    if rebuild:
        downloadtime = time.time()
        if not osmfile:
            osmfile = download_planet.downloadBBox(paths.osmdir + dbname + ".osm", latmin - 0.1, latmax + 0.1,
                                                   lonmin - 0.1, lonmax + 0.1)
        downloadtime = time.time() - downloadtime

        croploadtime = time.time()
        # cropLoadOSM.cropLoad(osmfile, dbname, latmin-0.1, latmax+0.1, lonmin-0.1, lonmax+0.1)
        database_generation_utils.load_db(osmfile, dbname)
        croploadtime = time.time() - croploadtime
    else:
        downloadtime = 0
        croploadtime = 0

    print("Times needed:")
    print("Download: {}s \ncropping and loading: {}s.".format(downloadtime, croploadtime))


def createPredictionFeatures(args):
    dbname = args.dbname
    latmin = args.latmin
    latmax = args.latmax
    lonmin = args.lonmin
    lonmax = args.lonmax
    osmfile = args.osmfile
    nWorkers = args.processors
    createDB(dbname, latmin, latmax, lonmin, lonmax, osmfile=osmfile, nWorkers=1, rebuild=args.rebuild)

    featuretime = time.time()
    fg = FeatureGenerator(dbname)
    fg.generateMap(latmin, latmax, lonmin, lonmax, granularity=0.001)
    fg.preproc_landuse_features_parallel(nWorkers)
    featuretime = time.time() - featuretime

    print("Time needed for the feature extraction: {}s".format(featuretime))
    return fg.saveFeatures()

def createFileFeatures(args):
    df = pd.read_csv(args.file)
    latmin = df.latitude.min()
    latmax = df.latitude.max()
    lonmin = df.longitude.min()
    lonmax = df.longitude.max()
    createDB(args.dbname, latmin, latmax, lonmin, lonmax, osmfile=args.osmfile, nWorkers=1, rebuild=args.rebuild)

    featuretime = time.time()
    fg = FeatureGenerator(args.dbname, filename=args.file, outpath="/")
    fg.set_data_from_pandas(df, value=args.value)
    fg.preproc_landuse_features_parallel(args.processors)
    featuretime = time.time() - featuretime

    print("Time needed for the feature extraction: {}s".format(featuretime))
    return fg.saveFeatures()

def createTrainFeatures():
    pass


def main(dbname, latmin, latmax, lonmin, lonmax, osmfile=None, nWorkers=1):
    featurefile = createPredictionFeatures(dbname, latmin, latmax, lonmin, lonmax, osmfile=osmfile, nWorkers=nWorkers)

def standardparsers(parser):
    parser.add_argument('-f', '--osmfile', type=str, help='path to .osm.pbf file if already present', default=None)
    parser.add_argument('-p', '--processors', type=int, help='Number of workers to use for parallel processes',
                        default=1)
    parser.add_argument('-r', '--rebuild', action='store_true', help='rebuild database')
    return parser

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
    parser_map.set_defaults(func=createPredictionFeatures)

    parser_file = subparsers.add_parser("file")
    parser_file.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser_file.add_argument("file", type=str, help="csv-file with lon, lat and value")
    parser_file.add_argument("-v", "--value", type=str, help="column of the value", default="value")
    parser_file = standardparsers(parser_file)
    parser_file.set_defaults(func=createFileFeatures)


    args = parser.parse_args()
    args.func(args)

