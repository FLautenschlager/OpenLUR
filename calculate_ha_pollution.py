import numpy as np


def calculate_ha_pollution(data, bounds, LAT=1, LON=2, IND_AVG_DATA=3):
    # Calculate average pollution per 100x100 tile
    pm_ha = []
    for x in range(bounds[0], bounds[1] + 1, 100):
        for y in range(bounds[2], bounds[3] + 1, 100):

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
    return pm_ha_numpy
