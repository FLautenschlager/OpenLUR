import argparse
import os.path
import time
from utils import paths

from OSM_featureExtraction import cropLoadOSM, download_planet
from OSM_featureExtraction.FeatureGenerator import FeatureGenerator

def createPredictionFeatures(dbname, latmin, latmax, lonmin, lonmax, osmfile=None, nWorkers=1):

    if not args.rebuild:
        if not checkDBexists(dbname):
            args.rebuild = True

    if args.rebuild:
        downloadtime = time.time()
        if not osmfile:
            osmfile = download_planet.downloadBBox(paths.osmdir + dbname + ".osm", latmin-0.1, latmax+0.1, lonmin-0.1, lonmax+0.1)
        downloadtime = time.time() - downloadtime

        croploadtime = time.time()
        #cropLoadOSM.cropLoad(osmfile, dbname, latmin-0.1, latmax+0.1, lonmin-0.1, lonmax+0.1)
        cropLoadOSM.loadDB(osmfile, dbname)
        croploadtime = time.time()-croploadtime
    else:
        downloadtime = 0
        croploadtime = 0


    featuretime = time.time()
    fg = FeatureGenerator(latmin, latmax, lonmin, lonmax, dbname)
    fg.preproc_landuse_features_parallel(nWorkers)
    featuretime = time.time() - featuretime

    print("Times needed:")
    print("Download: {}s \ncropping and loading: {}s \nfeature extraction: {}s".format(downloadtime, croploadtime, featuretime))
    return fg.saveFeatures()

def createTrainFeatures():
    pass

def main(dbname, latmin, latmax, lonmin, lonmax, osmfile=None, nWorkers=1):
    featurefile = createPredictionFeatures(dbname, latmin, latmax, lonmin, lonmax, osmfile=osmfile, nWorkers=nWorkers)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dbname', type=str, help='name of the DB (cityname)')
    parser.add_argument('latmin', type=float, help='minimum latitude')
    parser.add_argument('latmax', type=float, help='maximum latitude')
    parser.add_argument('lonmin', type=float, help='minimum longitude')
    parser.add_argument('lonmax', type=float, help='maximum longitude')
    #parser.add_argument('traindata', type=float, help='data to train classifier on')

    parser.add_argument('-f', '--osmfile', type=str, help='path to .osm.pbf file if already present', default=None)
    parser.add_argument('-p', '--processors', type=int, help='Number of workers to use for parallel processes', default=1)
    parser.add_argument('-r', '--rebuild', action='store_true', help='rebuild database')

    args = parser.parse_args()
    print(args.rebuild)
    main(args.dbname, args.latmin, args.latmax, args.lonmin, args.lonmax, osmfile=args.osmfile, nWorkers=args.processors)
