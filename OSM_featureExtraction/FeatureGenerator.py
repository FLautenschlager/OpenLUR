import argparse
import csv
import time
import numpy as np

import scipy.io as sio
from joblib import Parallel, delayed

from OSM_featureExtraction import Requestor
from utils import paths
from utils.wgs84_ch1903 import *


class FeatureGenerator:

    def __init__(self, dbname):
        self.dbname = dbname

    def __init__(self, latmin, latmax, lonmin, lonmax, dbname, granularity=0.001):
        self.generateMap(latmin, latmax, lonmin, lonmax, granularity=0.001)
        self.dbname = dbname

    def generateMap(self, latmin, latmax, lonmin, lonmax, granularity=0.001):
        self.data = []
        for lat in np.arange(latmin, latmax, granularity):
            for lon in np.arange(lonmin, lonmax, granularity):
                self.data.append([lat, lon])

    def setData(self, data):
        self.data = data

    def setCHdata(self, data):
        for row in data:
            x = row[0]
            y = row[1]
            row_new = [x, y, 0]
            lat = CHtoWGSlat(x + 50, y + 50)
            lon = CHtoWGSlng(x + 50, y + 50)
            row[0] = lat
            row[1] = lon
        self.data = data

    def preproc_landuse_features(self):
        data_new = []
        total_len = len(self.data)
        for i, row in enumerate(self.data):

            data_new.append(self.preproc_single_with_dist(row, self.dbname))

            if i % 100 == 0:
                print("{}".format(float(i) / total_len))

        self.data_with_features = data_new
        return data_new


    def preproc_landuse_features_parallel(self, n_workers=1):

        print("total rows: {}".format(len(self.data)))

        data_new = Parallel(n_jobs=n_workers, verbose=10)(delayed(self.preproc_single_with_dist)((row, self.dbname)) for row in self.data)
        self.data_with_features = data_new
        return data_new

    def preproc_single_with_dist(self, data):
        #print(data)
        row, city = data
        r = Requestor.Requestor(city)
        lat = row[0]
        lon = row[1]
        if len(row)==2:
            row_new = [lat, lon, 0]
        else:
            row_new = [lat, lon, row[2]]
        row_new.extend(r.create_features_withDist(lon, lat))

        return row_new

    def getDataWithFeatures(self):
        return self.data_with_features

    def saveFeatures(self):
        outfile = paths.lurdata + "{}_mapfeatures.csv".format(self.dbname)
        self.saveFeaturesToFile(outfile)
        return outfile

    def saveFeaturesToFile(self, outfile):
        with open(outfile, 'w') as myfile:
            wr = csv.writer(myfile)

            for row in self.data_with_features:
                wr.writerow(row)
            print("Done! File saved as {}.".format(myfile.name))


def main(latmin, latmax, lonmin, lonmax, database, nWorkers):


    fg = FeatureGenerator(latmin, latmax, lonmin, lonmax, database)
    fg.preproc_landuse_features_parallel(nWorkers)
    fg.saveFeatures()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("database", help="chose database, you previously made", type=str)
    parser.add_argument('latmin', type=float, help='minimum latitude')
    parser.add_argument('latmax', type=float, help='maximum latitude')
    parser.add_argument('lonmin', type=float, help='minimum longitude')
    parser.add_argument('lonmax', type=float, help='maximum longitude')
    parser.add_argument("-n", "--nWorkers", help="Number of parallel processes.")

    args = parser.parse_args()

    nWorkers = 1
    if args.nWorkers:
        nWorkers = int(args.nWorkers)

    main(args.latmin, args.latmax, args.lonmin, args.lonmax, args.database, nWorkers)
