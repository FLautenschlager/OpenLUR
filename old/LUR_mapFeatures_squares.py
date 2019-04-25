"""
Similar to LUR_mapFeatures.py but uses square buffers instead of regular round
buffers. This expects Lv03 (Swiss) coordinates as it is easier to deal with
squares when the coordinates are meters.
"""

import argparse
import time
import pandas as pd

import scipy.io as sio
from joblib import Parallel, delayed

from old import Requestor_squares as Requestor
from experiments import paths


def preproc_landuse_features_parallel(data, n_workers=1):
    func = preproc_single

    data_new = Parallel(n_jobs=n_workers, verbose=10)(
        delayed(func)(row) for row in data)
    return data_new


def preproc_single(row):
    r = Requestor.Requestor()

    lat = row[0] + 50
    lon = row[1] + 50
    row_new = [lat, lon]

    row_new.extend(r.create_features(lat, lon))

    row_new[0] -= 50
    row_new[1] -= 50

    return row_new


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nWorkers", type=int,
                        default=1, help="Number of parallel processes.")
    parser.add_argument("-sy", "--shift_y", type=int, default=0,
                        help="Meters to shift the original grid in y direction (east)")
    parser.add_argument("-sx", "--shift_x", type=int, default=0,
                        help="Meters to shift the original grid in x direction (north)")

    args = parser.parse_args()

    bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]

    # Shift bounds
    # bounds = [x + args.shift for x in bounds]
    bounds[0] = bounds[0] + args.shift_y
    bounds[1] = bounds[1] + args.shift_y
    bounds[2] = bounds[2] + args.shift_x
    bounds[3] = bounds[3] + args.shift_x

    data = []
    for y in range(bounds[0], bounds[1] + 1, 100):
        for x in range(bounds[2], bounds[3] + 1, 100):
            data.append([y, x])

    # data = [[684100, 247900]]
    # data = [[681000, 248000]]

    print("Starting feature generation. Total count of cells: {}".format(len(data)))
    start_time = time.time()
    data_new = preproc_landuse_features_parallel(data, n_workers=args.nWorkers)
    print("Features generated in {} minutes!".format(
        (time.time() - start_time) / 60))

    filenew = "mapfeatures_zurich_complete_shift_{}_{}.csv".format(
        str(args.shift_y), str(args.shift_x))

    data_new = pd.DataFrame(data_new)
    data_new.columns = ['y', 'x', 'commercial_100', 'commercial_200',
                        'commercial_300', 'industrial_100', 'industrial_200',
                        'industrial_300', 'residential_100', 'residential_200',
                        'residential_300', 'park_100', 'park_200', 'park_300',
                        'grass_100', 'grass_200', 'grass_300',
                        'water_100', 'water_200', 'water_300', 'roadtype_100',
                        'roadtype_200', 'roadtype_300', 'roaddist_motorway',
                        'roaddist_trunk', 'roaddist_primary',
                        'roaddist_secondary', 'roaddist_tertiary',
                        'roaddist_unclassified', 'roaddist_residential']
    # data_new.columns = ['y', 'x', 'roadtype_100', 'roadtype_200', 'roadtype_300', 'roaddist_motorway',
    #                     'roaddist_trunk', 'roaddist_primary', 'roaddist_secondary', 'roaddist_tertiary']

    data_new.to_csv(filenew, index=False)

    print("Done! File saved as {}.".format(filenew))
