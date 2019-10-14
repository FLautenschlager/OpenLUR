import argparse
import csv
import pickle
from os import listdir, mkdir
from os.path import join, isfile, isdir

import pandas as pd

from Models.GAM_featureSelection import GAM_featureSelection
from experiments import paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--seasonNumber", help="Number of season", type=int, default=1)
    parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int, default=1)
    parser.add_argument("-p", "--processes", help="Number of parallel processes", type=int, default=1)

    args = parser.parse_args()

    seasons = ["pm_ha_ext_01042012_30062012", "pm_ha_ext_01072012_31092012",
               "pm_ha_ext_01102012_31122012", "pm_ha_ext_01012013_31032013"]

    file = seasons[args.seasonNumber - 1] + '_landUse_withDistances.csv'
    feat = "OSM_land_use_distances"

    feat_columns = ["commercial{}m".format(i) for i in range(50, 3050, 50)]
    feat_columns.extend(["industrial{}m".format(i) for i in range(50, 3050, 50)])
    feat_columns.extend(["residential{}m".format(i) for i in range(50, 3050, 50)])

    feat_columns.extend(["bigStreet{}m".format(i) for i in range(50, 1550, 50)])
    feat_columns.extend(["localStreet{}m".format(i) for i in range(50, 1550, 50)])

    feat_columns.extend(
        ["distanceTrafficSignal", "distanceMotorway", "distancePrimaryRoad", "distanceIndustrial"])

    target = 'pm_measurement'

    directory = join(paths.featuresel,
                     "season{}_Features_{}_r2val_{}iterations".format(args.seasonNumber, feat, args.iterations))
    if not isdir(directory):
        mkdir(directory)

    filelist = listdir(directory)

    filenumber = []
    for f in filelist:
        if isfile(f):
            try:
                filenumber.append(int(f[:-2]))
            except TypeError:
                pass

    if len(filenumber) == 0:
        startNumber = 0
    else:
        startNumber = max(filenumber) + 1

    data = []
    with open(paths.lurdata + file, 'r') as myfile:
        reader = csv.reader(myfile)
        for row in reader:
            data.append([float(i) for i in row])

    data = pd.DataFrame(data)
    col_total = ['x', 'y', target]
    col_total.extend(feat_columns)
    data.columns = col_total
    print(data.columns)

    dataset = args.seasonNumber

    if args.iterations:
        iterations = args.iterations
    else:
        iterations = 5

    if args.processes:
        njobs = args.processes
    else:
        njobs = 1

    # Feature selection:
    features_total = []
    rmse_total = []
    r2_total = []
    r2val_total = []
    i = 0
    makeIt = True

    for i in range(iterations):
        selector = GAM_featureSelection(njobs=njobs, verbosity=0)
        final_features, r2, rmse, pvalues = selector.select_features(data, feat_columns[:], target)
        # gam = GAM(njobs=njobs, niter=iterations, verbosity=1)
        # rmse, r2 = gam.test_model(data, final_features, target)

        file = join(directory, "{}.p".format(i))
        pickle.dump({'rmse': rmse, 'r2': r2, 'features': final_features, 'pvalues': pvalues},
                    open(file, 'wb'))

        print("Saved in file {}".format(file))