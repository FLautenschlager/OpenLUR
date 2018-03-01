"""
Script which manages a number of experiments across different models.
"""

from os.path import expanduser, join, isfile
import subprocess
from multiprocessing.dummy import Pool
import argparse
import pandas as pd

RESULTS_PATH = expanduser(join('~', 'lur_results_earlystop.json'))


def start_process(job):
    print('Curjob:', job)
    print('Start:', job['regressor_path'],
          job['calib_file'], job['feature_cols'], job['interpolation_factor'])
    # Start a process for a single cross validation
    subprocess.check_call(['python3', job['regressor_path'],
                           job['calib_file'], RESULTS_PATH,
                           '-f', str(job['feature_cols']),
                           '-i', str(job['interpolation_factor'])])
    print('End:', job['regressor_path'],
          job['calib_file'], job['feature_cols'], job['interpolation_factor'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desired_runs', type=int,
                        help='Number of runs that are desired per data point')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of concurrent jobs')
    args = parser.parse_args()

    regressor_paths = {
        # 'gam': join('Hasenfratz', 'gam_r_parallel.py'),
        # 'lasso': join('Hasenfratz', 'lasso.py'),
        # 'mean': join('Hasenfratz', 'mean.py'),
        'nn': expanduser(join('/mnt', 'c', 'Users', 'DerGuteste', 'hg', 'environment-net', 'landuse_regression.py'))
        # 'nn': expanduser(join('~', 'hg', 'environment-net', 'landuse_regression.py'))
    }

    calib_files = {
        '01042012_30062012': expanduser(join('~', 'data', 'opensense',
                                             'pm_01042012_30062012_filtered_ha_200.csv')),
        '01072012_31092012': expanduser(join('~', 'data', 'opensense',
                                              'pm_01072012_31092012_filtered_ha_200.csv')),
        '01102012_31122012': expanduser(join('~', 'data', 'opensense',
                                             'pm_01102012_31122012_filtered_ha_200.csv')),
        '01012013_31032013': expanduser(join('~', 'data', 'opensense',
                                             'pm_01012013_31032013_filtered_ha_200.csv'))
    }
    # calib_files = {
    #     '01042012_30062012': expanduser(join('~', 'data', 'opensense', 'flo_features',
    #                                          'pm_ha_ext_01042012_30062012_landUse.csv')),
    #     '01072012_31092012': expanduser(join('~', 'data', 'opensense', 'flo_features',
    #                                           'pm_ha_ext_01072012_31092012_landUse.csv')),
    #     '01102012_31122012': expanduser(join('~', 'data', 'opensense', 'flo_features',
    #                                          'pm_ha_ext_01102012_31122012_landUse.csv')),
    #     '01012013_31032013': expanduser(join('~', 'data', 'opensense', 'flo_features',
    #                                          'pm_ha_ext_01012013_31032013_landUse.csv'))
    # }

    # Hasenfratz feature columns
    feature_cols = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                    'streetsize', 'traffic_tot', 'streetdist_l']
    # Flo's feature columns
    # feature_cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239']

    interpolation_factors = {
        '0': 0.00#,
        # '0.25': 0.25,
        # '0.5': 0.5,
        # '0.75': 0.75,
        # '1.0': 1.0
    }

    todo_index_tuples = [(source, timeframe)
                            for source in regressor_paths
                            for timeframe in calib_files]
                            # for interpolation_factor in interpolation_factors]

    todo_index = pd.MultiIndex.from_tuples(
        todo_index_tuples, names=['source', 'timeframe'])

    todo = pd.Series(args.desired_runs, index=todo_index)


    # See which runs are missing
    if isfile(RESULTS_PATH):
        # Look at what results do already exist and see which runs are missing
        results_table = pd.read_json(RESULTS_PATH)
        # print(results_table)

        done = results_table.groupby(['source', 'timeframe']).size()
        todo = todo.subtract(done, fill_value=0).apply(int)
        todo = todo[todo > 0]

    print(todo)

    # Firstly, fill job queue
    job_queue = []
    for ind, runs_todo in todo.iteritems():
        # ind[0] = source; ind[1] = timeframe; ind[2] = interpolation_factor

        reg_path = regressor_paths[ind[0]]
        calib_file = calib_files[ind[1]]
        interpolation_factor = 0 # interpolation_factors[ind[2]]

        # Queue as many jobs of this type as necessary
        for _ in range(runs_todo):
            job_queue.append({'calib_file': calib_file,
                              'feature_cols': feature_cols,
                              'regressor_path': reg_path,
                              'interpolation_factor': interpolation_factor})

    # Work all jobs
    with Pool(args.jobs) as pool:
        pool.map(start_process, job_queue)
