"""
Utility functions specific to Hasenfratz' data set.
"""

import pandas as pd
import scipy.io as sio

def load_input_file(input_file_path):
    """ Load .csv or .mat input file """

    file_extension = input_file_path[-4:]

    if file_extension == '.mat':
        # Load data
        pm_ha = sio.loadmat(input_file_path)['pm_ha']

        # Prepare data
        data_1 = pd.DataFrame(pm_ha[:, :3])
        data_2 = pd.DataFrame(pm_ha[:, 7:])
        calib_data = pd.concat([data_1, data_2], axis=1)
        calib_data.columns = ['x', 'y', 'pm_measurement', 'population', 'industry', 'floorlevel', 'heating', 'elevation', 'streetsize',
                              'signaldist', 'streetdist', 'slope', 'expo', 'traffic', 'streetdist_m', 'streetdist_l', 'trafficdist_l', 'trafficdist_h', 'traffic_tot']

        return calib_data

    elif file_extension == '.csv':
        # Load data
        return pd.read_csv(input_file_path)

    else:
        print('Invalid file extension: ', file_extension)