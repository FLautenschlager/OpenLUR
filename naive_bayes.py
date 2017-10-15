from os.path import expanduser
import scipy.io as sio
import math
import numpy as np
import csv

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import itertools

def preproc_land_use(filename):
	data = sio.loadmat(filename)['data']

	IND_AVG_DATA = 3
	GEO_ACC = 4
	print("Shape before cleaning: ", data.shape)
	data = data[data[:, GEO_ACC] < 3, :]  # Remove inaccurate data
	data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
	data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

	bounds = sio.loadmat(rootdir + "Shared/UFP_Delivery_Lautenschlager/matlab/bounds")['bounds']

	print("Shape after cleaning: ", data.shape)

	LAT = 1
	LON = 2
	pm_ha = []
	for x in range(bounds[0, 0], bounds[0, 1], 100):
		for y in range(bounds[0, 2], bounds[0, 3], 100):

			# Fetch data in the bounding box
			temp = data[
			       (data[:, LAT] >= x) & (data[:, LAT] < (x + 100)) & (data[:, LON] >= y) & (data[:, LON] < (y + 100)),
			       :]
			if temp.shape[0] != 0:
				# Calculate Statistics and dependent variable
				m = np.mean(temp[:, IND_AVG_DATA])

				pm_ha.append([x, y, m])

	del data

	LAT = 0
	LON = 1

	print("Training shape with tram depots: ({}, {})".format(len(pm_ha), len(pm_ha[1])))

	# Tram depots
	ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500], [681700, 249500], [683700, 251500],
	             [679400, 248500],
	             [683400, 249900], [683400, 249800], [682500, 243400]]

	# check if Tram depot in bounding box
	pm_ha_numpy = np.array(pm_ha)
	for depot in ha_depots:
		pm_ha_numpy = pm_ha_numpy[~((pm_ha_numpy[:, LAT] == depot[LAT]) & (pm_ha_numpy[:, LON] == depot[LON])), :]

	pm_ha = pm_ha_numpy.tolist()
	# print(pm_ha_numpy[:, 9:].max())

	del pm_ha_numpy

	return pm_ha


def classification(data):
	model = GaussianNB()

	data_np = np.array(data)
	scores = cross_val_score(model, data_np[:, 0:2], data_np[:, 3], cv=5)
	print(scores)
	y_pred = model.predict(data_np[:, 0:2])
	cnf = confusion_matrix(data_np[:,3], y_pred)
	plt.figure()
	plot_confusion_matrix(cnf, [0,10000,20000,30000,40000,60000], title='confusion matrix, without normalization')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
	rootdir = expanduser("~/Data/OpenSense/")
	datadir = rootdir + "Shared/UFP_Delivery_Lautenschlager/matlab/data/seasonal_maps/filt/"
	file = "pm_01072012_31092012_filtered.mat"

	data = preproc_land_use(datadir + file)

	with open(rootdir + file[:-4] + "_naive_bayes.csv", 'w') as myfile:
		wr = csv.writer(myfile)
		print(myfile.name)

		for row in data:
			wr.writerow(row)

#	with open(rootdir + file[:-4] + "_naive_bayes.csv", 'r') as myfile:
#		reader = csv.reader(myfile)
#		print(myfile.name)
#		data = []
#		for row in reader:
#			data.append([float (i) for i in row].append(int(float(row[2])/10000)))

	for row in data:
		row.append(	with open(rootdir + file[:-4] + "_naive_bayes.csv", 'r') as myfile:
		reader = csv.reader(myfile)
		print(myfile.name)
		data = []
		for row in reader:
			data.append(int(float(row[2])/10000))

	classification(data)
