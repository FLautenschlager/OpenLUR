"""
Utility functions specific to Hasenfratz' data set.
"""

from os.path import isfile

import pandas as pd
import scipy.io as sio


def load_input_file(input_file_path):
    """ Load .csv or .mat input file """

    file_extension = input_file_path.split('.')[-1]

    if file_extension == 'mat':
        # Load data
        pm_ha = sio.loadmat(input_file_path)['pm_ha']

        # Prepare data
        data_1 = pd.DataFrame(pm_ha[:, :3])
        data_2 = pd.DataFrame(pm_ha[:, 7:])
        calib_data = pd.concat([data_1, data_2], axis=1)
        calib_data.columns = ['x', 'y', 'pm_measurement', 'population', 'industry', 'floorlevel', 'heating', 'elevation', 'streetsize',
                              'signaldist', 'streetdist', 'slope', 'expo', 'traffic', 'streetdist_m', 'streetdist_l', 'trafficdist_l', 'trafficdist_h', 'traffic_tot']

        return calib_data

    elif file_extension == 'csv':
        # Load data
        return pd.read_csv(input_file_path)

    else:
        print('Invalid file extension: ', file_extension)


def write_results_file(output_file_path, results):
    """ Write results into a file either as json or csv """

    file_extension = output_file_path.split('.')[-1]

    # Convert feature_cols to string so that they stay together
    results['feature_cols'] = str(results['feature_cols'])
    
    results = pd.DataFrame(results, index=[0])


    if file_extension == 'json':

        # Read old results file if it exists
        if isfile(output_file_path):
            old_results = pd.read_json(output_file_path)

            # Combine old and new results (even if they have different columns)
            results = pd.concat(
                [old_results, results], axis=0, ignore_index=True)

        # Write combined results to file and retry indefinitely if it failed
        while True:
            try:
                results.to_json(output_file_path)
            except:
                continue
            break

    elif file_extension == 'csv':
        # The initial write has to write the column headers if the file doesn't
        # exist yet
        initial_write = not isfile(output_file_path)

        # Write result to file and retry indefinitely if it failed
        while True:
            try:
                results.to_csv(
                    output_file_path, mode='a', header=initial_write, index=False)
            except:
                continue
            break

    else:
        print('Invalid file extension: ', file_extension)

def is_in(train_cell, test_data):
    """Test whether a train_cell overlaps with any cell in the test data"""
    train_y = train_cell['y']
    train_x = train_cell['x']
    for cell in test_data.itertuples():
        test_y = cell.y
        test_x = cell.x
        if train_y >= test_y - 100 and train_y <= test_y + 100 and train_x >= test_x - 100 and train_x <= test_x + 100:
            return True

    return False
    