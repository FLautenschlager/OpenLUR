import sys
if "/home/florian/Code/code-2017-land-use" not in sys.path:
	sys.path.append("/home/florian/Code/code-2017-land-use")

from utils import paths
import csv
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import argparse


def cross_validation(X_t, y_t, layers, nodes):
	kf = KFold(n_splits=10, shuffle=True)
	score = []
	score_train = []
	rmse = []
	mae = []
	batch_size = 4096 * 4

	for train, test in kf.split(X_t):
		X_train, y_train = X_t[train, :], y_t[train]
		X_test, y_test = X_t[test, :], y_t[test]
		model = Sequential()


		model.add(Dense(20, activation='relu', input_shape=(240,)))
		for _ in range(layers-1):
			model.add(Dense(nodes,activation='relu'))
		# model.add(Dropout(0.1, name='dropout_1'))

		model.add(Dense(1))
		model.compile(loss=r2_keras, optimizer='adam')
		model.fit(X_train, y_train, batch_size=batch_size, epochs=10000, validation_data=(X_test, y_test), verbose=0)

		pred = model.predict(X_test, verbose=0)

		score_train.append(1 - model.evaluate(X_train, y_train, verbose=0))
		score.append(1 - model.evaluate(X_test, y_test, verbose=0))
		mae.append(mean_absolute_error(y_test, pred))
		rmse.append(mean_squared_error(y_test, pred))

	scoreMean = np.mean(score)
	# print("R2-score on training folds = " + score_train)
	# print("R2-score on test folds = " + score)
	print("Mean R2-score on training data = {}".format(np.mean(score_train)))
	print("Mean R2-score on test data = {}".format(np.mean(scoreMean)))
	return scoreMean, np.mean(mae), np.sqrt(np.mean(rmse))


def r2_keras(y_true, y_pred):
	SS_res = K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return SS_res / (SS_tot + K.epsilon())


def adj_r2(y_true, y_pred):
	n = tf.cast(K.shape(y_true)[0], dtype=tf.float32)

	k = tf.cast(241.0, dtype=tf.float32)

	counter = tf.divide(tf.reduce_sum(tf.subtract(y_true, y_pred)), tf.subtract(n, k))
	noun = tf.reduce_sum(tf.subtract(y_true, tf.reduce_mean(y_true))) / tf.subtract(n, 1)

	return tf.divide(counter, noun)


def r2(y_true, y_pred):
	first = tf.reduce_sum(tf.subtract(y_true, y_pred))
	second = tf.reduce_sum(tf.subtract(y_true, tf.reduce_mean(y_true)))
	r = tf.divide(first, second)
	return r


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help="Define an input file.")
	parser.add_argument("-l", "--layers", help="Count of hidden layers", type=int)
	parser.add_argument("-n", "--nodes", help="Number of nodes in each hidden layer", type=int)
	parser.add_argument("-i", "--iterations", help="Number of iterations to mean on", type=int)

	args = parser.parse_args()



	if args.file:
		file = args.file
	else:
		file = "pm_ha_ext_01012013_31032013_landUse.csv"
		# file = "pm_ha_ext_01042012_30062012_landUse.csv"
		# file = "pm_ha_ext_01072012_31092012_landUse.csv"
		# file = "pm_ha_ext_01102012_31122012_landUse.csv"

	if args.layers:
		layers = args.layers
	else:
		layers = 4

	if args.nodes:
		nodes = args.nodes
	else:
		nodes = 20

	if args.iterations:
		iterations = args.iterations
	else:
		iterations = 40

	data = []
	with open(paths.lurdata + file, 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	train_data_np = np.array(data)

	X_train, y_train = train_data_np[:, 3:], train_data_np[:, 2]

	X_train = np.ascontiguousarray(X_train)
	y_train = np.ascontiguousarray(y_train)

	r2_total = []
	mae_total = []
	rmse_total = []
	for _ in range(iterations):
		r2, mae, rmse = cross_validation(X_train, y_train, layers, nodes)
		r2_total.append(r2)
		mae_total.append(mae)
		rmse_total.append(rmse)

	print("R2 = {}\nMAE = {}\nRMSE = {}".format(np.mean(r2_total), np.mean(mae_total), np.mean(rmse_total)))
