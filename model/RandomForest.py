import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from model.AbstractModel import AbstractModel
from utils.DataLoader import Dataset
from utils.MyPool import MyPool as Pool


class RandomForestRandomSearch(AbstractModel):

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor
        self.m = self.model()

    def param_search(self, x, y, iterations=60, score_rank_lowest=False, processes=1,
                     **kwargs):

        modeldicts = [self.random_modeldict() for i in range(iterations)]

        standarddict = {
            'bootstrap': True,
            'max_features': 'auto',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 10
        }

        modeldicts.append(standarddict)

        results = pd.DataFrame([self.param_search_para(m, x, y, processes) for m in modeldicts])

        results.columns = ['rmse', 'mae', 'r2', 'model_dict']
        results.sort_values("r2", ascending=False, inplace=True)
        self.m = self.model(**results.iloc[0]["model_dict"])

        best_model = results.iloc[0]

        self.log.debug("Best Model: score: {:.2f}".format(results.iloc[0].r2))
        return self.concat_results(best_model['rmse'], best_model['mae'], best_model['r2'], best_model["model_dict"])

    def param_search_para(self, modeldict, x, y, processes):

        kf = KFold(n_splits=10, shuffle=True)
        rmse = []
        mae = []
        r2 = []
        for train_index, test_index in kf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m = self.model(**modeldict)
            m.fit(X_train, y_train)
            predictions = m.predict(X_test)
            rmse_iter, mae_iter, r2_iter = self.score_function(y_test, predictions)
            rmse.append(rmse_iter)
            mae.append(mae_iter)
            r2.append(r2_iter)

        self.log.debug("{:135}: {: 6.4f}".format("{}".format(modeldict), np.mean(r2)))
        return np.mean(rmse), np.mean(mae), np.mean(r2), modeldict

    def param_search_iteration(self, inputs):
        x, y, modeldict = inputs
        modeldict["n_jobs"] = 4
        kf = KFold(n_splits=10, shuffle=True)
        rmse = []
        mae = []
        r2 = []
        for train_index, test_index in kf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m = self.model(**modeldict)
            m.fit(X_train, y_train)
            predictions = m.predict(X_test)
            rmse_iter, mae_iter, r2_iter = self.score_function(y_test, predictions)
            rmse.append(rmse_iter)
            mae.append(mae_iter)
            r2.append(r2_iter)

        self.log.debug("{:135}: {: 6.4f}".format("{}".format(modeldict), np.mean(r2)))
        return np.mean(rmse), np.mean(mae), np.mean(r2), modeldict

    @staticmethod
    def random_modeldict():
        temp = np.random.choice(["auto", float(np.random.uniform(0., 1., size=1))])
        try:
            temp = float(temp)
        except ValueError:
            pass

        return {
            "n_estimators": int(np.random.uniform(1, 1000, size=1)),
            "min_samples_leaf": int(np.random.uniform(1, 100, size=1)),
            "min_samples_split": int(np.random.uniform(2, 20, size=1)),
            "bootstrap": bool(np.random.choice([True, False], size=1)),
            "max_features": temp
        }


class RandomForestStandard(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor
        self.m = self.model()

    def param_search(self, x, y, **kwargs):
        kf = KFold(n_splits=10, shuffle=True)
        rmse = []
        mae = []
        r2 = []
        for train_index, test_index in kf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.m = self.model()
            self.m.fit(X_train, y_train)
            predictions = self.m.predict(X_test)
            rmse_iter, mae_iter, r2_iter = self.score_function(y_test, predictions)
            rmse.append(rmse_iter)
            mae.append(mae_iter)
            r2.append(r2_iter)

        print("{}: {}".format(self.m.get_params(), np.mean(r2)))

        self.log.info("Reached a RMSE of {}, MAE of {} and R2 of {}.".format(np.mean(rmse), np.mean(mae), np.mean(r2)))

        return self.concat_results(np.mean(rmse), np.mean(mae), np.mean(r2))


if __name__ == "__main__":
    data, feat_columns, target, feat = Dataset.OpenSenseOSM1()
    x = data[feat_columns].values
    y = data[target].values

    R = RandomForestRandomSearch()
    res = R.param_search(x, y)
    print(res)
