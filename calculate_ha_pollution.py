import numpy as np
import pandas as pd
import paths
import argparse
import scipy.io as sio
import csv
from clean_data import clean_data


def calculate_ha_pollution(data, bounds, LAT=1, LON=2, IND_AVG_DATA=3):
    # Calculate average pollution per 100x100 tile
    pm_ha = []
    for x in range(bounds[0], bounds[1] + 1, 100):
        for y in range(bounds[2], bounds[3] + 1, 100):

            # Fetch data in the bounding box
            temp = data[(data[:, LAT] >= x) & (data[:, LAT] < (x + 100))
                        & (data[:, LON] >= y) & (data[:, LON] < (y + 100)), :]

            if temp.shape[0] == 0:
                pm_num = [x, y, 0, 0, 0, 0, 0, 0]

            else:

                # Calculate Statistics and dependent variable
                m = np.mean(temp[:, IND_AVG_DATA])
                s = np.std(temp[:, IND_AVG_DATA])
                med = np.median(temp[:, IND_AVG_DATA])

                log = np.log(temp[:, IND_AVG_DATA])
                # log[log == -float('Inf')] = 0
                log = log[log != -float('Inf')]

                m_log = np.mean(log)
                s_log = np.std(log)

                pm_num = [x, y, m, temp.shape[0], s, m_log, s_log, med]

            pm_ha.append(pm_num)

    pm_ha_numpy = np.array(pm_ha)
    return pm_ha_numpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input file')
    args = parser.parse_args()

    data = None

    bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]

    if args.input_file[-4:] == '.mat':
        data = sio.loadmat(args.input_file)['data']
        cleaned_data = clean_data(data)
    elif args.input_file[-4:] == '.csv':
        data = pd.read_csv(args.input_file)
        cleaned_data = clean_data(data.iloc, IND_AVG_DATA=data.columns.get_loc(
            'particleNumber'), IND_GEO_ACC=data.columns.get_loc('GPSPrecision'))

    else:
        print('File type "{ft}" not supported'.format(ft=args.input_file[-4:]))

    # Use default columns for LAT, LON and IND_AVG_DATA
    pm_ha_data = calculate_ha_pollution(cleaned_data, bounds)

    # Put data in a pandas DataFrame
    pm_ha_data_df = df.DataFrame(pm_ha_data)
    pm_ha_data_df.columns = ['x', 'y', 'pm_measurement',
                             'pm_measurement_length', 'pm_std', 'pm_mean_log',
                             'pm_std_log', 'pm_median', 'population',
                             'industry', 'floorlevel', 'heating',
                             'elevation', 'streetsize', 'signaldist',
                             'streetdist', 'slope', 'expo', 'traffic',
                             'streetdist_m', 'streetdist_l',
                             'trafficdist_l', 'trafficdist_h',
                             'traffic_tot']

    # Cut the '.mat' from the mat's filename and append '_ha.csv'
    new_filename = args.input_file[:-4] + '_ha.csv'

    # Write file
    pm_ha_data_df.to_csv(new_filename)
