import argparse
import math
from os.path import join

import numpy as np
import pandas as pd
import scipy.io as sio

from experiments import paths

BOUNDS_PATH = join(paths.rootdir, 'bounds')
MODEL_VAR_PATH = join(paths.rdir, 'model_ha_variables.mat')
OUTPUT_COLS = ['y', 'x', 'pm_measurement', 'population',
               'industry', 'floorlevel', 'heating',
               'elevation', 'streetsize', 'signaldist',
               'streetdist', 'slope', 'expo', 'traffic',
                             'streetdist_m', 'streetdist_l',
                             'trafficdist_l', 'trafficdist_h',
                             'traffic_tot']


def clean_data(data):

    data = data[data['GPSPrecision'] < 3]  # Remove inaccurate data
    data = data[data['particleNumber'] < math.pow(10, 5)]  # Remove outliers

    # Hasenfratz doesn't remove 0-values here so we don't do that either for now
    # data = data[data['particleNumber'] != 0]  # Remove 0-values

    return data


def calculate_ha_pollution(data, bounds):
    # Calculate average pollution per 100x100 tile
    pm_ha = []
    for y in range(bounds[0], bounds[1] + 1, 100):
        for x in range(bounds[2], bounds[3] + 1, 100):

            # Fetch data in the bounding box
            # temp = data.where(data['latitude'] >= x).where(data['latitude'] < (x + 100.0)).where(data['longitude'] >= y).where(data['longitude'] < (y + 100.0))

            # Note: Each coordinate is specified by its south-western corner
            # Swiss coordinates have x and y, x increases to the north,
            # y increases to the east
            temp = data[((data['x'] >= x) & (data['x'] < (x + 100.0))
                         & (data['y'] >= y) & (data['y'] < (y + 100.0)))]
            # print(bounds)
            # print(x + 100)
            # print(data['latitude'])
            #
            # print(data['latitude'] < (x + 100.0))
            #             # & (data['longitude'] >= y) & (data['longitude'] < (y + 100.0)))
            # print(temp)
            # print(temp.shape)

            if temp.shape[0] == 0:
                pm_num = [y, x, 0, 0, 0, 0, 0, 0]

            else:

                # Calculate Statistics and dependent variable
                m = np.mean(temp['particleNumber'])
                s = np.std(temp['particleNumber'])
                med = np.median(temp['particleNumber'])

                log = np.log(temp['particleNumber'])
                # log[log == -float('Inf')] = 0
                log = log[log != -float('Inf')]

                m_log = np.mean(log)
                s_log = np.std(log)

                pm_num = [y, x, m, temp.shape[0], s, m_log, s_log, med]

            pm_ha.append(pm_num)

    pm_ha_pd = pd.DataFrame(pm_ha)
    pm_ha_pd.columns = ['y', 'x', 'pm_measurement', 'pm_measurement_length',
                        'pm_std', 'pm_mean_log', 'pm_std_log', 'pm_median']
    pm_ha_pd.set_index(['y', 'x'])
    return pm_ha_pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('-r', '--rows', type=int,
                        help='Number of rows', default=200)
    parser.add_argument('-dr', '--default_rows', action='store_true',
                        help='Output several files with a range of row numbers')
    args = parser.parse_args()

    # Load bounds
    bounds = sio.loadmat(BOUNDS_PATH)['bounds'][0]

    # Load model variables
    model_var = pd.DataFrame(sio.loadmat(MODEL_VAR_PATH)['model_variables'])
    model_var.columns = ['y', 'x', 'population', 'industry', 'floorlevel',
                         'heating', 'elevation', 'streetsize', 'signaldist',
                         'streetdist', 'slope', 'expo', 'traffic',
                         'streetdist_m', 'streetdist_l', 'trafficdist_l',
                         'trafficdist_h', 'traffic_tot']
    model_var.set_index(['y', 'x'])

    # Assume .csv
    data = pd.read_csv(args.input_file)
    cleaned_data = clean_data(data)

    # Use default columns for LAT, LON and IND_AVG_DATA
    pm_ha_data = calculate_ha_pollution(cleaned_data, bounds)
    print(pm_ha_data)

    # Tram depots
    ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500],
                 [681700, 249500], [683700, 251500], [679400, 248500],
                 [683400, 249900], [683400, 249800], [682500, 243400]]

    # check if Tram depot in bounding box
    for depot in ha_depots:
        pm_ha_data = pm_ha_data[~((pm_ha_data['y'] == depot[0]) &
                                  (pm_ha_data['x'] == depot[1]))]

    # Remove zeros
    pm_ha_data = pm_ha_data[pm_ha_data['pm_measurement'] != 0]
    pm_ha_data = pm_ha_data[pm_ha_data['pm_std'] != 0]

    # Merge pm_ha_data and model_var and select relevant columns
    pm_ha_data = model_var.merge(pm_ha_data)[['y', 'x', 'pm_measurement',
                                              'pm_measurement_length', 'pm_std', 'pm_mean_log',
                                              'pm_std_log', 'pm_median', 'population',
                                              'industry', 'floorlevel', 'heating',
                                              'elevation', 'streetsize', 'signaldist',
                                              'streetdist', 'slope', 'expo', 'traffic',
                                              'streetdist_m', 'streetdist_l',
                                              'trafficdist_l', 'trafficdist_h',
                                              'traffic_tot']]

    pm_ha_data = pm_ha_data.sort_values(
        by='pm_measurement_length', ascending=False)[OUTPUT_COLS]

    print(pm_ha_data)

    if args.default_rows:
        pm_ha_data_list = [
            pm_ha_data.iloc[:1200],
            pm_ha_data.iloc[:1100],
            pm_ha_data.iloc[:1000],
            pm_ha_data.iloc[:900],
            pm_ha_data.iloc[:800],
            pm_ha_data.iloc[:700],
            pm_ha_data.iloc[:600],
            pm_ha_data.iloc[:500],
            pm_ha_data.iloc[:400],
            pm_ha_data.iloc[:300],
            pm_ha_data.iloc[:250],
            pm_ha_data.iloc[:200],
            pm_ha_data.iloc[:170],
            pm_ha_data.iloc[:140],
            pm_ha_data.iloc[:110],
            pm_ha_data.iloc[:80],
            pm_ha_data.iloc[:50],
            pm_ha_data.iloc[:20]
        ]

        for data in pm_ha_data_list:
            # Cut the '.mat' from the mat's filename and append '_ha.csv'
            new_filename = args.input_file[:-4] + '_ha_' + str(len(data)) + '.csv'
            data.to_csv(new_filename, index=False)

    else:

        # Get the args.rows tiles with most measurements
        pm_ha_data = pm_ha_data.iloc[:args.rows]
        
        # Cut the '.mat' from the mat's filename and append '_ha.csv'
        new_filename = args.input_file[:-4] + '_ha_' + str(args.rows) + '.csv'
        
        # Write file
        pm_ha_data.to_csv(new_filename, index=False)
