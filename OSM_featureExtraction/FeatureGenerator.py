import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from OSM_featureExtraction import OSMRequestor
from utils.wgs84_ch1903 import *


class FeatureGenerator:

    def __init__(self, dbname, filename=None, outpath=None):
        self.dbname = dbname
        self.filename = dbname + ".csv"
        if filename:
            self.filename = filename

        self.outpath = ""
        if outpath:
            self.outpath = outpath

        self.featuremethods = [self.getStandardFeatures]

    def generateMap(self, latmin, latmax, lonmin, lonmax, granularity=0.001):
        self.data = []
        for lat in np.arange(latmin, latmax, granularity):
            for lon in np.arange(lonmin, lonmax, granularity):
                self.data.append([lat, lon])

    def setData(self, data):
        self.data = data

    def set_data_from_pandas(self, df, lon="longitude", lat="latitude", value="value"):
        self.data = list(df[[lat, lon, value]].values)

    def set_data_from_file(self, file, lon="longitude", lat="latitude", value="value"):
        df = pd.read_csv(file)
        return self.set_data_from_pandas(df, lon=lon, lat=lat, value=value)

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

            data_new.append(self.preproc_single(row))

            if i % 100 == 0:
                print("{}".format(float(i) / total_len))

        self.data_with_features = pd.DataFrame(data_new)
        return data_new

    def preproc_landuse_features_parallel(self, n_workers=1):

        print("Number of rows for the feature extraction: {}".format(len(self.data)))

        data_new = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(self.preproc_single)(row) for row in self.data)
        self.data_with_features = pd.DataFrame(data_new)
        return data_new

    def preproc_single(self, row):
        lat = row[0]
        lon = row[1]
        if len(row) == 2:
            row_new = {"latitude": lat, "longitude": lon, "target": 0}
        else:
            row_new = {"latitude": lat, "longitude": lon, "target": row[2]}
        [row_new.update(m(lat, lon)) for m in self.featuremethods]

        return row_new

    def getStandardFeatures(self, lat, lon):
        r = OSMRequestor.Requestor(self.dbname)
        return r.create_features(lon, lat)

    def getDataWithFeatures(self):
        return self.data_with_features

    def saveFeatures(self):
        outfile = self.outpath + "{}_mapfeatures.csv".format(self.filename[:-4])
        print(outfile)
        self.saveFeaturesToFile(outfile)
        return outfile

    def saveFeaturesToFile(self, outfile):
        self.data_with_features.to_csv(outfile, index=False)
        print("Done! File saved as {}.".format(outfile))

    def add_featuremethod(self, featuremethod):
        self.featuremethods.append(featuremethod)

def main(database, file, n_workers):
    fg = FeatureGenerator(database)
    fg.set_data_from_file(file, value="conct")
    fg.preproc_landuse_features_parallel(n_workers)
    fg.saveFeatures()


if __name__ == "__main__":
    ## ****************************** password is "docker" ****************************

    parser = argparse.ArgumentParser()

    parser.add_argument("database", help="chose database, you previously made", type=str)
    parser.add_argument("file", help="file to build features for", type=str)
    parser.add_argument("-n", "--nWorkers", help="Number of parallel processes.")

    args = parser.parse_args()

    nWorkers = 1
    if args.nWorkers:
        nWorkers = int(args.nWorkers)

    main(args.database, args.file, args.nWorkers)
