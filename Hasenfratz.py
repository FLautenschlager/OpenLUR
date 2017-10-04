from os.path import expanduser
import scipy.io as sio
import math
import numpy as np


def feature(feat, row, col):
	LAT = 0
	LON = 1
	# Subtract 1 from column because of matlab (index starts with 1)
	n = feat[(feat[:, LAT] == row[LAT]) & (feat[:, LON] == row[LON]), col - 1]
	if len(n) == 0:
		n = 0
	return n


def feature_sum(feat, row, col):
	LAT = 0
	LON = 1
	ind = ((feat[:, LAT] == row[LAT]) & (feat[:, LON] == row[LON]))
	n = []
	# temp = feat[ind,:]
	for c in col:
		# n.append(temp[c])
		n.append(feat[ind, c])
	if len(n) == 0:
		n = 0
	n = np.sum(n)
	return n


rootdir = expanduser("~/Data/OpenSense/Shared/UFP_Delivery_Lautenschlager/matlab/")
datadir = rootdir + "data/seasonal_maps/"
data = sio.loadmat(datadir + "filt/pm_01072012_31092012_filtered.mat")['data']

IND_AVG_DATA = 3
GEO_ACC = 4
print("Shape before cleaning: ", data.shape)
data = data[data[:, GEO_ACC] < 3, :]  # Remove inaccurate data
data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

bounds = sio.loadmat(rootdir + "bounds")['bounds']

print("Shape after cleaning: ", data.shape)

LAT = 1
LON = 2
pm_ha = []
for x in range(bounds[0, 0], bounds[0, 1], 100):
	for y in range(bounds[0, 2], bounds[0, 3], 100):

		# Fetch data in the bounding box
		temp = data[(data[:, LAT] >= x) & (data[:, LAT] < (x + 100)) & (data[:, LON] >= y) & (data[:, LON] < (y + 100)),:]
		if temp.shape[0] != 0:

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

del data

LAT = 0
LON = 1

print("Training shape with tram depots: ({}, {})".format(len(pm_ha), len(pm_ha[1])))

# Tram depots
ha_depots = [[681800, 247400], [681700, 247400], [681700, 247500], [681700, 249500], [683700, 251500], [679400, 248500],
             [683400, 249900], [683400, 249800], [682500, 243400]]

# check if Tram depot in bounding box
pm_ha_numpy = np.array(pm_ha)
for depot in ha_depots:
	pm_ha_numpy = pm_ha_numpy[~((pm_ha_numpy[:, LAT] == depot[LAT]) & (pm_ha_numpy[:, LON] == depot[LON])), :]

pm_ha = pm_ha_numpy.tolist()

del pm_ha_numpy


# Load land use data
pop = sio.loadmat(rootdir + 'landuse_data/population_zh_2011.mat')['pop']
indus = sio.loadmat(rootdir + 'landuse_data/industry_zh_2008.mat')['indus']
floor_level = sio.loadmat(rootdir + 'landuse_data/floorlevel_zh_2011.mat')['floor_level']
heating_oil_gas = sio.loadmat(rootdir + 'landuse_data/heatinggasoil_zh_2011.mat')['heating_oil_gas']
elev = sio.loadmat(rootdir + 'landuse_data/elevation_zh.mat')['elev']
street_size_max = sio.loadmat(rootdir + 'landuse_data/streetsize_max_zh.mat')['street_size_max']
street_length = sio.loadmat(rootdir + 'landuse_data/streetlength_zh.mat')['street_length']
dist2street = sio.loadmat(rootdir + 'landuse_data/dist_to_street.mat')['dist2street']
dist2signal = sio.loadmat(rootdir + 'landuse_data/dist_to_signal.mat')['dist2signal']
dist2traffic = sio.loadmat(rootdir + 'landuse_data/dist_to_traffic_ugz.mat')['dist2traffic']
slope_exp = sio.loadmat(rootdir + 'landuse_data/slope_exp_zh.mat')['slope_exp']
traffic = sio.loadmat(rootdir + 'landuse_data/traffic_zh_2007.mat')['traffic']
traffic_ugz = sio.loadmat(rootdir + 'landuse_data/traffic_ugz_2013.mat')['traffic_ugz']

# Append land use data to pm data
print("Training shape without tram depots: ({}, {})".format(len(pm_ha), len(pm_ha[1])))
for row in pm_ha:
	row.append(feature(pop, row, 2))
	row.append(feature_sum(indus, row, [2, 3, 4]))
	row.append(feature(floor_level, row, 2))
	row.append(feature(heating_oil_gas, row, 2))
	row.append(feature(elev, row, 2))
	row.append(feature(street_size_max, row, 2))
	row.append(feature(dist2signal, row, 2))
	row.append(feature(dist2street, row, 2))
	row.append(feature(slope_exp, row, 2))
	row.append(feature(slope_exp, row, 3))
	row.append(feature_sum(traffic, row, [10, 11]))
	row.append(feature(dist2street, row, 3))
	row.append(feature(dist2street, row, 4))
	row.append(feature(dist2traffic, row, 2))
	row.append(feature(dist2traffic, row, 3))
	row.append(feature(traffic_ugz, row, 5))

print("Training shape with features: ({}, {})".format(len(pm_ha), len(pm_ha[1])))


# Simple GAM
# todo: Cross validation, other .mat files
print("Training model")
pm_ha_numpy = np.array(pm_ha)

from pygam import LinearGAM
from pygam.utils import generate_X_grid

gam = LinearGAM(n_splines=10).gridsearch(pm_ha_numpy[:,-16:], pm_ha_numpy[:,2])
XX = generate_X_grid(gam)

gam.summary()
