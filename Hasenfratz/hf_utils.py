"""
Utility functions specific to Hasenfratz' data set.
"""

from os.path import isfile
import portalocker

import pandas as pd
import scipy.io as sio

from scipy.interpolate import griddata
import numpy as np
import random


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

        # For json we have to deal with the concurrent file access therefore
        # i use portalocker to lock the file during reading, constructing the
        # new json, and writing
        with portalocker.Lock(output_file_path, mode='a+') as f:

            f.seek(0)

            # Read old results file if it exist
            if f.read(1) != '':
                f.seek(0)
                old_results = pd.read_json(f)

                # Delete old content
                f.seek(0)
                f.truncate()

                # Combine old and new results (even if they have different columns)
                results = pd.concat(
                    [old_results, results], axis=0, ignore_index=True)

            # Write combined results to file and retry indefinitely if it failed
            results.to_json(f)


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

# INTERPOLATION STUFF

FEATURE_COLS = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                'streetsize', 'traffic_tot', 'streetdist_l', 'population',
                'heating', 'signaldist', 'streetdist', 'traffic',
                'streetdist_m', 'trafficdist_l', 'trafficdist_h']
LABEL_COL = 'pm_measurement'

# Find random point in triangle
def uniform_sample_triangle(p_a, p_b, p_c):
    # see http://www.cs.princeton.edu/~funk/tog02.pdf section 4.2

    p_a = np.array([float(p_a['y']), float(p_a['x'])])
    p_b = np.array([float(p_b['y']), float(p_b['x'])])
    p_c = np.array([float(p_c['y']), float(p_c['x'])])

    r_1 = random.uniform(0.0,1.0)
    r_2 = random.uniform(0.0,1.0)

    new_p = (1-np.sqrt(r_1))*p_a + (np.sqrt(r_1)*(1-r_2))*p_b + (r_2*np.sqrt(r_1))*p_c

    return {'y': new_p[0], 'x': new_p[1]}

# interpolate in triangle
def barycentric_interpolation(p_a, p_b, p_c, new_point, col):
    # see: https://codeplea.com/triangular-interpolation
    y_a = float(p_a['y'])
    y_b = float(p_b['y'])
    y_c = float(p_c['y'])
    y_new = float(new_point['y'])
    x_a = float(p_a['x'])
    x_b = float(p_b['x'])
    x_c = float(p_c['x'])
    x_new = float(new_point['x'])

    denominator = (y_b-y_c)*(x_a-x_c)+(x_c-x_b)*(y_a-y_c)
    w_a = ((y_b-y_c)*(x_new-x_c)+(x_c-x_b)*(y_new-y_c))/denominator
    w_b = ((y_c-y_a)*(x_new-x_c)+(x_a-x_c)*(y_new-y_c))/denominator
    w_c = 1 - w_a - w_b

    return w_a*float(p_a[col]) + w_b*float(p_b[col]) + w_c*float(p_c[col])

# Interpolate between two points
def linear_interpolation(p_a, p_b, new_point, col):
    y_a = float(p_a['y'])
    y_b = float(p_b['y'])
    y_new = float(new_point['y'])
    x_a = float(p_a['x'])
    x_b = float(p_b['x'])
    x_new = float(new_point['x'])

    # Calculate distance between p_a and p_b
    vec_a_b = [y_b-y_a, x_b-x_a]
    dist = np.linalg.norm(vec_a_b)

    # Calculate distance between p_a and new_point
    vec_a_new = [y_new-y_a, x_new-x_a]
    dist_new = np.linalg.norm(vec_a_new)

    # Interpolate a function over the distance between the points
    # For that we redefine what x and y are:
    # dist = 0 -> label of p_a, dist = max(dist) -> label of p_b
    # In the following: x = dist, y = label
    x_a = 0
    y_a = float(p_a[col])
    x_b = dist
    y_b = float(p_b[col])
    x_new = dist_new

    # see: https://en.wikipedia.org/wiki/Linear_interpolation
    return (y_a*(x_b-x_new) + y_b*(x_new-x_a))/(x_b-x_a)

# Interpolate a point with the appropriate interpolation method
def interpolate_candidate(can, col, new_point, method='linear'):
    # Note: it is not needed to check south-west tiles because candidates are
    # generated in a way that guarantees that those tiles are not empty

    tile_coords = [[float(can['sw']['y']), float(can['sw']['x'])]]
    tile_labels = [float(can['sw'][col])]
    tiles = [can['sw']]

    if not can['se'].empty:
        tiles.append(can['se'])
        tile_coords.append([float(can['se']['y']), float(can['se']['x'])])
        tile_labels.append(float(can['se'][col]))
    if not can['ne'].empty:
        tiles.append(can['ne'])
        tile_coords.append([float(can['ne']['y']), float(can['ne']['x'])])
        tile_labels.append(float(can['ne'][col]))
    if not can['nw'].empty:
        tiles.append(can['nw'])
        tile_coords.append([float(can['nw']['y']), float(can['nw']['x'])])
        tile_labels.append(float(can['nw'][col]))

    # print(tile_coords, tile_labels)

    if len(tiles) == 4:
        return griddata(tile_coords, tile_labels, [float(new_point['y']), float(new_point['x'])], method=method)[0]
    elif len(tiles) == 3:
        return barycentric_interpolation(tiles[0], tiles[1], tiles[2], new_point, col)
    elif len(tiles) == 2:
        return linear_interpolation(tiles[0], tiles[1], new_point, col)
    else:
        print('something went wrong', tiles)

def find_candidates(data):
    candidates = []

    # Find all instances of 4 tiles with labels in a square
    for south_west_tuple in data.itertuples():
        # Note: x and y specify the south-western corner of a tile
        x = south_west_tuple.x
        y = south_west_tuple.y
        south_west_row = data.loc[(data.x==x)&(data.y==y)]
        north_west_row = data.loc[(data.x==x+100.0)&(data.y==y)]
        north_east_row = data.loc[(data.x==x+100.0)&(data.y==y+100.0)]
        south_east_row = data.loc[(data.x==x)&(data.y==y+100.0)]
        if len(north_east_row) > 0 or len(south_east_row) > 0 or len(north_west_row) > 0:
            candidates.append({
                'nw': north_west_row,
                'ne': north_east_row,
                'se': south_east_row,
                'sw': south_west_row})

    # print(candidates)
    print('candidates_len', len(candidates))
    return candidates

# Generate interpolated rows from a list of interpolation candidates
def generate_rows(candidates, number):
    generated_rows = []
    # Generate new data points that lie in a random four tile square
    for _ in range(number):
        # Randomly choose a four tile square
        can = random.sample(candidates, 1)[0]

        # Randomly choose the south western coordinates for a new data point
        # All 4 tiles exist
        if not can['se'].empty and not can['nw'].empty and not can['ne'].empty:
            new_x = random.uniform(float(can['sw']['x']), float(can['nw']['x']))
            new_y = random.uniform(float(can['sw']['y']), float(can['se']['y']))
        # ne missing
        elif not can['se'].empty and not can['nw'].empty and can['ne'].empty:
            new_point = uniform_sample_triangle(can['sw'], can['se'], can['nw'])
            new_x = new_point['x']
            new_y = new_point['y']
        # nw missing
        elif not can['se'].empty and can['nw'].empty and not can['ne'].empty:
            new_point = uniform_sample_triangle(can['sw'], can['se'], can['ne'])
            new_x = new_point['x']
            new_y = new_point['y']
        # se missing
        elif can['se'].empty and not can['nw'].empty and not can['ne'].empty:
            new_point = uniform_sample_triangle(can['sw'], can['ne'], can['nw'])
            new_x = new_point['x']
            new_y = new_point['y']
        # ne + nw missing
        elif not can['se'].empty and can['nw'].empty and can['ne'].empty:
            new_x = float(can['sw']['x'])
            new_y = random.uniform(float(can['sw']['y']), float(can['se']['y']))
        # ne + se missing
        elif can['se'].empty and not can['nw'].empty and can['ne'].empty:
            new_x = random.uniform(float(can['sw']['x']), float(can['nw']['x']))
            new_y = float(can['sw']['y'])
        # nw + se missing
        elif can['se'].empty and can['nw'].empty and not can['ne'].empty:
            offset = random.uniform(0.0,100.0)
            new_x = float(can['sw']['x']) + offset
            new_y = float(can['sw']['y']) + offset
        else:
            print('something is wrong', can)

        new_row = {'x': new_x, 'y': new_y}


        # Interpolate all features and the label
        new_row[LABEL_COL] = interpolate_candidate(can, LABEL_COL, new_row)
        for feature_col in FEATURE_COLS:
            new_row[feature_col] = interpolate_candidate(can, feature_col, new_row)

        # print('candidate:', can)
        # print('new_row:', new_row)

        # Mark row as interpolated
        new_row['interpolated'] = True
        generated_rows.append(new_row)

    return generated_rows

# Interpolate 'number' rows from 'data'
def interpolate(data, number):

    if number <= 0:
        return data

    # Reset index so that y and x can be easily adressed
    data = data.reset_index()

    # Mark all existing points as not interpolated
    data['interpolated'] = False

    # Find candidates in data that could be interpolated
    candidates = find_candidates(data)

    # Generate interpolated rows from candidates
    generated_rows = generate_rows(candidates, number)

    generated_rows_df = pd.DataFrame(generated_rows)
    all_data = pd.concat([data, generated_rows_df])

    all_data = all_data.set_index(['y', 'x'])

    return all_data
