import pandas as pd
import scipy.io as sio
import csv
from utils import paths

def loadData(seasonNumber, features):
    seasons = ["pm_ha_ext_01042012_30062012", "pm_ha_ext_01072012_31092012",
               "pm_ha_ext_01102012_31122012", "pm_ha_ext_01012013_31032013"]

    if features == 1:
      file = seasons[seasonNumber - 1] + '.mat'
      feat = "OpenSense"
      pm_ha = sio.loadmat(paths.extdatadir + file)['pm_ha']
      # Prepare data
      data_1 = pd.DataFrame(pm_ha[:, :3])
      data_2 = pd.DataFrame(pm_ha[:, 7:])
      data = pd.concat([data_1, data_2], axis=1)
      data.columns = ["x", "y", "pm_measurement", "population", "industry", "floorlevel", "heating",
                      "elevation", "streetsize",
                      "signaldist", "streetdist", "slope", "expo", "traffic", "streetdist_m", "streetdist_l",
                      "trafficdist_l", "trafficdist_h", "traffic_tot"]

      feat_columns = ['industry', 'floorlevel', 'elevation', 'slope', 'expo', 'streetsize', 'traffic_tot',
                      'streetdist_l']
      target = 'pm_measurement'

    elif (features == 2) | (features == 3):
      file = seasons[seasonNumber - 1] + '_landUse.csv'
      feat = "OSM_land_use"

      feat_columns = ["commercial{}m".format(i) for i in range(50, 3050, 50)]
      feat_columns.extend(["industrial{}m".format(i) for i in range(50, 3050, 50)])
      feat_columns.extend(["residential{}m".format(i) for i in range(50, 3050, 50)])

      feat_columns.extend(["bigStreet{}m".format(i) for i in range(50, 1550, 50)])
      feat_columns.extend(["localStreet{}m".format(i) for i in range(50, 1550, 50)])

      target = 'pm_measurement'

      if features == 3:
        file = file[:-4] + "_withDistances.csv"
        feat = feat + "_distances"
        feat_columns.extend(
          ["distanceTrafficSignal", "distanceMotorway", "distancePrimaryRoad", "distanceIndustrial"])

      data = []
      with open(paths.lurdata + file, 'r') as myfile:
        reader = csv.reader(myfile)
        for row in reader:
          data.append([float(i) for i in row])

      data = pd.DataFrame(data)
      col_total = ['x', 'y']
      col_total.append(target)
      col_total.extend(feat_columns)
      data.columns = col_total
    print(data.columns)

    return data, feat_columns, target, feat
