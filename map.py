import os
from os.path import expanduser
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from cartopy import config
import cartopy.crs as ccrs

rootdir = expanduser(
    "~\\hg\\code-2017-land-use\\UFP_Delivery_Lautenschlager\\matlab\\")
datadir = rootdir + "data\\seasonal_maps\\"
data = sio.loadmat(datadir + "filt\\pm_01042012_30062012_filtered.mat")['data']

IND_AVG_DATA = 3
GEO_ACC = 4
print("Shape before cleaning: ", data.shape)
data = data[data[:, GEO_ACC] < 3, :]  # Remove inaccurate data
data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

print("Shape after cleaning: ", data.shape)

bounds = sio.loadmat(rootdir + "bounds")['bounds']
x_len = int((bounds[0][1] - bounds[0][0]) / 100 + 1)
y_len = int((bounds[0][3] - bounds[0][2]) / 100 + 1)

# Create average pollution per 100x100 tile
LAT = 1
LON = 2
pm_ha = []
for x in range(bounds[0, 0], bounds[0, 1] + 1, 100):
    for y in range(bounds[0, 2], bounds[0, 3] + 1, 100):

        # Fetch data in the bounding box
        temp = data[(data[:, LAT] >= x) & (data[:, LAT] < (x + 100))
                    & (data[:, LON] >= y) & (data[:, LON] < (y + 100)), :]

        if temp.shape[0] == 0:
            pm_num = [x, y, 0, 0, 0, 0, 0, 0]

        else:

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

pm_ha_numpy = np.array(pm_ha)

pollution = pm_ha_numpy[:, 2]
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
