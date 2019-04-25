import argparse
import logging

from experiments.paramsearch_experiments.AutoML import autoML

logger = logging.getLogger('AutoML')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--seasonNumber", help="Number of season", type=int, default=1)
	parser.add_argument("-f", "--features",
						help="Dataset to build model on: (1) OpenSense, (2) OSM, (3) OSM + distances", type=int,
						default=1)
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int, default=1)
	parser.add_argument("-p", "--processes", help="Number of parallel processes", type=int, default=4)
	parser.add_argument("-t", "--time", help="Give time for parametertuning of each fold in seconds", type=int,
						default=60)
	parser.add_argument("-r", "--refit", help="determines, if the model should be refittet", action='store_true')

	args = parser.parse_args()

	autoML(args.seasonNumber, args.features, args.iterations, args.time, args.processes, args.refit)
