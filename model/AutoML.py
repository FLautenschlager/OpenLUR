from model.AbstractModel import AbstractModel
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import KFold


class AutoML(AbstractModel):

    def __init__(self):
        super().__init__()
        self.model = AutoSklearnRegressor

    def fit(self, x, y, modeldict=None):
        if not self.m:
            self.param_search(x, y)
        self.m.refit(x, y)

    def param_search(self,
                     x,
                     y,
                     time=600, #18000,
                     **kwargs):
        self.m = AutoSklearnRegressor(
            time_left_for_this_task=time,
            resampling_strategy="cv",
            resampling_strategy_arguments={'folds': 10}
        )

        self.m.fit(x, y, metric=mean_squared_error, dataset_name="Land Use Regression")
        # print(self.m.sprint_statistics())
        # score = score_funtion(y, self.m.predict(x))
        # print("Reached a score of {}.".format(score))

        kf = KFold(n_splits=10, shuffle=True)
        rmse = []
        mae = []
        r2 = []
        for train_index, test_index in kf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.m.refit(X_train, y_train)
            predictions = self.m.predict(X_test)
            rmse_iter, mae_iter, r2_iter = self.score_function(y_test, predictions)
            rmse.append(rmse_iter)
            mae.append(mae_iter)
            r2.append(r2_iter)

        # print("Reached a RMSE of {}, MAE of {} and R2 of {}.".format(np.mean(rmse), np.mean(mae), np.mean(r2)))

        return self.concat_results(np.mean(rmse), np.mean(mae), np.mean(r2))
