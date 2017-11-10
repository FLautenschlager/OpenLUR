import sys

sys.path.append("/home/florian/Code/code-2017-land-use/")
#sys.path.append("/home/florian/Code/code-2017-land-use/HeatMapBcc/HeatMapBCC/python")

import csv
import paths
import scipy.io as sio
#from HeatMapBcc.HeatMapBCC.python.heatmapbcc import HeatMapBCC
from HeatMapBCC.python import heatmapbcc
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

def preprocessDataForHeatMapBCC(data):
	dataNew = []

	treshold = 13000
	for row in data:
		if row[2]<(treshold-2000):
			dataNew.append([1, row[0], row[1], 0])
		elif row[2]>(treshold+2000):
			dataNew.append([1, row[0], row[1], 2])
		else:
			dataNew.append([1, row[0], row[1], 1])

	return dataNew


if __name__ == "__main__":
	file = "pm_01072012_31092012_filtered.mat"

	data = []
	with open(paths.lurdata + file[:-4] + "_landUse.csv", 'r') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			data.append([float(i) for i in row])

	data = np.array(preprocessDataForHeatMapBCC(data))
	bounds = sio.loadmat(paths.rootdir + "bounds")['bounds']
	nx = (bounds[0, 0] - bounds[0, 1]) / 100
	ny = (bounds[0, 2] - bounds[0, 3]) / 100

	alpha0 = np.ones((3,3,1))
	model = heatmapbcc.HeatMapBCC(nx, ny, 3, 3, alpha0, 1, z0=[0.5,0.5,0.5])

	model.combine_classifications(data)

	E_t_out, kappa_out, v_kappa_out = model.predict(data[:,1], data[:,2])

	predictions = []
	print(E_t_out.shape)
	for i in range(E_t_out.shape[1]):
		predictions.append(np.argmax(E_t_out[:,i]))

	predictions = np.array(predictions)

	print(predictions.shape)
	print(data[:,3].shape)

	#print(confusion_matrix(data[:,3], predictions))

	pickle.dump({'model': model, 'data':data, 'predictions':predictions},
	            open(paths.bayesiandata + file[:-4] + "_bayes.p", 'wb'))