import pickle
from glob import iglob

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from model.AutoML import AutoML
from model.GAM import GAM
from model.RandomForest import RandomForestRandomSearch, RandomForestStandard

models = {
    "Random_Forest_random_search": RandomForestRandomSearch,
    "AutoML": AutoML,
    "Random_Forest_Standard": RandomForestStandard,
    "GAM": GAM
}


class RegressionRunner:

    def __init__(self, model, modelname="defaultname", tensorboard=None, iteration=0):
        self.model = model
        self.modelname = modelname
        self.tensorboard = tensorboard
        self.iteration = iteration
        if self.tensorboard is not None:
            self.tb_exists = True
        else:
            self.tb_exists = False

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def param_search(self, X, y, **kwargs):
        results = self.model.param_search(X, y, **kwargs)
        label = "search"
        self.tensorboard.add_scalar("{}_r2".format(label), results["r2"], self.iteration)
        self.tensorboard.add_scalar("{}_rmse".format(label), results["rmse"], self.iteration)
        self.tensorboard.add_scalar("{}_mae".format(label), results["mae"], self.iteration)
        return results

    def evaluate(self, target, prediction, label="undefined"):
        result = pd.Series()
        result['rmse'] = np.sqrt(mean_squared_error(target, prediction))
        result['mae'] = mean_absolute_error(target, prediction)
        result['r2'] = r2_score(target, prediction)
        self.tensorboard.add_scalar("{}_r2".format(label), result["r2"], self.iteration)
        self.tensorboard.add_scalar("{}_rmse".format(label), result["rmse"], self.iteration)
        self.tensorboard.add_scalar("{}_mae".format(label), result["mae"], self.iteration)
        # print("The model {} achieved: \n\t RMSE: {}\n\t MAE:  {}\n\t R2:   {}".format(self.modelname, result['rmse'], result['mae'], result['r2']))
        return result

    def run(self, x_train, y_train, x_test=None, y_test=None):

        search_results = self.param_search(x_train, y_train, processes=10)
        search_results['type'] = "search"
        self.train(x_train, y_train)
        if x_test.any():
            y_pred = self.predict(x_test)
            train_results = self.evaluate(y_train, self.predict(x_train), label="train")
            test_results = self.evaluate(y_test, y_pred, label="test")
            if self.tb_exists:
                title = "r2: {:5.3f}; rmse: {:5.3f}; mae: {:5.3f}".format(test_results["r2"], test_results["rmse"], test_results["mae"])
                self.add_plot_to_summary(self.plot_predictions(y_test, y_pred, title=title), self.iteration, "Test_prediction")
                self.add_plot_to_summary(self.plot_errors(y_test, y_pred, title=title), self.iteration, "Test_error")
            train_results["type"] = "train"
            test_results["type"] = "test"
            results = pd.DataFrame([search_results, train_results, test_results])
        else:
            train_results = None
            test_results = None
            results = pd.DataFrame([search_results])

        return results

    def plot_predictions(self, y, y_pred, title=None):
        fig = plt.figure()
        plt.plot(y_pred, label='Predicted')
        plt.plot(y, label="True")
        ticks = np.linspace(0, len(y), 5).tolist()
        plt.xticks(ticks)
        plt.legend(loc='upper left')
        if title:
            plt.title(title)
        return fig

    def plot_errors(self, y, y_pred, title=None):
        fig = plt.figure()
        plt.plot(np.abs(y_pred - y))
        ticks = np.linspace(0, len(y), 5).tolist()
        plt.xticks(ticks)
        if title:
            plt.title(title)
        return fig

    def add_plot_to_summary(self, plt, epoch, title):
        self.tensorboard.add_figure(title, plt, epoch)


def run_regression(modelname, x_train, y_train, x_test, y_test, iteration=0, filename=None, tensorboard=None):
    runner = RegressionRunner(models[modelname](), modelname=modelname, iteration=iteration, tensorboard=tensorboard)
    results = runner.run(x_train, y_train, x_test, y_test)
    if filename:
        pickle.dump((results), open(filename, "wb"))
    return results


def test_londondata(model):
    path = "../data/laei/2013/CSV/pm10_split_central_nonjoined/mapfeatures/*"

    train = []
    test = None
    for file in iglob(path):
        if "test" in file:
            test = pd.read_csv(file)
        train.append(pd.read_csv(file))
    train = pd.concat(train, ignore_index=True)
    cols = list(train.columns)
    for col in cols:
        d = train[col].dtypes
        if d != "float64":
            print(col)
            print(d)
    # print(len(cols))
    cols.remove("target")
    cols.remove("latitude")
    cols.remove("longitude")
    # cols.remove("traffic_signals")
    # cols.remove("industrial")
    # cols.remove("motorway")
    # cols.remove("primary")
    # print(len(cols))

    x_train = train[cols].values
    x_test = test[cols].values
    y_train = train["target"].values
    y_test = test["target"].values

    run_regression(model, x_train, y_train, x_test, y_test, "laei_{}_iterations_{}.p".format(model))


def test_opensense(model):
    path = "../data/OpenSense/seasonal_maps/lur/pm_01072013_31092013_filtered_ha_landUse.csv"

    data = pd.read_csv(path, header=None)
    # print(data.head())
    # print(data.shape)
    cols = [i + 3 for i in range(data.shape[1] - 3)]
    # print(len(cols))
    # print(data[cols].head())
    train = data
    test = data

    x_train = train[cols].values
    x_test = test[cols].values
    y_train = train[2].values
    y_test = test[2].values

    # r = RegressionRunner("AutoML_self")
    # m = AutoSklearnRegressor(time_left_for_this_task=300)
    # m.fit(x_train, y_train, dataset_name="OpenSense")
    # print(m.show_models())
    # pred = m.predict(x_test)
    # r.evaluate(y_test, pred)

    run_regression(model, x_train, y_train, x_test, y_test, "opensense_{}_iterations_{}.p".format(model))


if __name__ == "__main__":
    model = "Random_Forest_random_search"
    model = "AutoML"
    test_londondata(model)
