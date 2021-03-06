import pandas as pd
import scipy.io as sio
from glob import iglob

seasons = ["pm_ha_ext_01042012_30062012", "pm_ha_ext_01072012_31092012",
           "pm_ha_ext_01102012_31122012", "pm_ha_ext_01012013_31032013"]

rootpath = "data/"
defaultpathOSM = rootpath + "OpenSenseOSM/"
defaultpathOSO = rootpath + "OpenSenseOriginal/"
defaultpathLAEI = rootpath + "laeiOSM/"

class Dataset:

    @staticmethod
    def OpenSenseOSM1(path=defaultpathOSM):
        return Dataset.OpenSenseOSM(1, path=path)

    @staticmethod
    def OpenSenseOSM2(path=defaultpathOSM):
        return Dataset.OpenSenseOSM(2, path=path)

    @staticmethod
    def OpenSenseOSM3(path=defaultpathOSM):
        return Dataset.OpenSenseOSM(3, path=path)

    @staticmethod
    def OpenSenseOSM4(path=defaultpathOSM):
        return Dataset.OpenSenseOSM(4, path=path)

    @staticmethod
    def OpenSenseOSM(season=1, path=defaultpathOSM):
        file = path + seasons[season - 1] + '_OSM.csv'
        data = pd.read_csv(file)

        cols = list(data.columns)
        for col in cols:
            d = data[col].dtypes
            if d != "float64":
                print(col)
                print(d)
        cols.remove("target")
        try:
            cols.remove("x")
            cols.remove("y")
        except ValueError:
            try:
                cols.remove("latitude")
                cols.remove("longitude")
            except ValueError:
                pass

        x_train = data[cols]
        y_train = data["target"].values/1000

        x_test = None
        y_test = None

        return x_train, y_train, x_test, y_test

    @staticmethod
    def OpenSenseOriginal1(path=defaultpathOSO):
        return Dataset.OpenSenseOriginal(1, path=path)

    @staticmethod
    def OpenSenseOriginal2(path=defaultpathOSO):
        return Dataset.OpenSenseOriginal(2, path=path)

    @staticmethod
    def OpenSenseOriginal3(path=defaultpathOSO):
        return Dataset.OpenSenseOriginal(3, path=path)

    @staticmethod
    def OpenSenseOriginal4(path=defaultpathOSO):
        return Dataset.OpenSenseOriginal(4, path=path)

    @staticmethod
    def OpenSenseOriginal(season=1, path=defaultpathOSO):
        file = path + seasons[season - 1] + '.mat'

        pm_ha = sio.loadmat(file)['pm_ha']
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

        x_train = data[feat_columns]
        y_train = data[target].values/1000

        x_test = None
        y_test = None

        return x_train, y_train, x_test, y_test

    @staticmethod
    def laeiOSM(path=defaultpathLAEI):
        train = []
        test = None
        for file in iglob(path + "*"):
            if "test" in file:
                test = pd.read_csv(file)
            else:
                d = pd.read_csv(file)
                train.append(d)
        train = pd.concat(train, ignore_index=True)
        cols = list(train.columns)
        for col in cols:
            d = train[col].dtypes
            if d != "float64":
                print(col)
                print(d)
        cols.remove("target")
        cols.remove("latitude")
        cols.remove("longitude")

        x_train = train[cols]
        x_test = test[cols]
        y_train = train["target"].values
        y_test = test["target"].values

        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x, y, x_test, y_test = Dataset.laeiOSM(path="../data/laeiOSM/")

    print(x.shape)
    print(y.shape)
    try:
        print(x_test.shape)
        print(y_test.shape)
    except:
        pass

    print(x[0])
