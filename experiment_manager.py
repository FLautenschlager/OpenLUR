"""
Script which manages a number of experiments across different models.
"""

from os.path import expanduser, join, isfile
import subprocess
from multiprocessing.dummy import Pool
import argparse
import pandas as pd

RESULTS_PATH = expanduser(join('~', 'lur_results.json'))


def start_process(job):
    print('Curjob:', job)
    print('Start:', job['regressor_path'],
          job['calib_file'], job['feature_cols'])
    # Start a process for a single cross validation
    subprocess.check_call(['python3', job['regressor_path'],
                           job['calib_file'], RESULTS_PATH,
                           '-f', str(job['feature_cols'])])
    print('End:', job['regressor_path'],
          job['calib_file'], job['feature_cols'])


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

    feature_cols = ['industry', 'floorlevel', 'elevation', 'slope', 'expo',
                    'streetsize', 'traffic_tot', 'streetdist_l']


    todo_index_tuples = [(source, timeframe)
                            for source in regressor_paths
                            for timeframe in calib_files]

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
        # ind[0] = source; ind[1] = timeframe

        reg_path = regressor_paths[ind[0]]
        calib_file = calib_files[ind[1]]

        # Queue as many jobs of this type as necessary
        for _ in range(runs_todo):
            job_queue.append({'calib_file': calib_file,
                              'feature_cols': feature_cols,
                              'regressor_path': reg_path})

    # Work all jobs
    with Pool(args.jobs) as pool:
        pool.map(start_process, job_queue)
