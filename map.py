import paths
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from calculate_ha_pollution import calculate_ha_pollution
from clean_data import clean_data

LAT = 1
LON = 2
IND_AVG_DATA = 3
GEO_ACC = 4

# Load necessary files
data = sio.loadmat(paths.filtereddatadir + 'pm_01042012_30062012_filtered.mat')['data']
bounds = sio.loadmat(paths.rootdir + 'bounds')['bounds'][0]

# Clean data
print('Shape before cleaning: ', data.shape)
data = clean_data(data, IND_AVG_DATA=IND_AVG_DATA, GEO_ACC=GEO_ACC)
print('Shape after cleaning: ', data.shape)

# Calculate average pollution per 100x100 tile
pm_ha_numpy = calculate_ha_pollution(data, bounds, LAT=LAT, LON=LON, IND_AVG_DATA=IND_AVG_DATA)

pollution = pm_ha_numpy[:, 2]

# Calculate the amount of tiles in x and y direction respectively
x_len = int((bounds[1] - bounds[0]) / 100 + 1)
y_len = int((bounds[3] - bounds[2]) / 100 + 1)

# Arrange pollution data array into a matrix that corresponds to the tiles in
# which the city was devided into
pollution = np.reshape(pollution, (x_len, y_len))
# Not quite sure yet why i have to rotate it but it works
pollution = np.rot90(pollution)

# Plot pollution data
plt.imshow(pollution, extent=[0, x_len, 0, y_len])
plt.axis('off')
plt.colorbar()
plt.show()
