import paths
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio

# Load necessary files
# data = sio.loadmat(join(paths.filtereddatadir, 'pm_01042012_30062012_filtered.mat'))['data']
data = pd.read_csv(join(paths.extdatadir, 'pm_01072012_31092012_filtered_ha_200.csv'))
bounds = sio.loadmat(join(paths.rootdir, 'bounds'))['bounds'][0]

data = data.set_index(['x', 'y'])

# data is sparse (tiles without label do not exist)
# make it dense for easier map creation
dense_data = []
for x in range(bounds[0], bounds[1] + 1, 100):
    for y in range(bounds[2], bounds[3] + 1, 100):
        dense_data.append(data.loc[[(float(x), float(y))]])

dense_data_df = pd.concat(dense_data)

# Calculate the amount of tiles in x and y direction respectively
x_len = int((bounds[1] - bounds[0]) / 100 + 1)
y_len = int((bounds[3] - bounds[2]) / 100 + 1)

# Replace NaN labels with 0
dense_data_df = dense_data_df.fillna(value=0)
print(dense_data_df)

# Arrange pollution data array into a matrix that corresponds to the tiles in
# which the city was devided into
pollution = np.reshape(dense_data_df['pm_measurement'], (x_len, y_len))
# Not quite sure yet why i have to rotate it but it works
pollution = np.rot90(pollution)

# Plot pollution data
plt.imshow(pollution, extent=[0, x_len, 0, y_len])
plt.axis('off')
plt.colorbar()
plt.show()
