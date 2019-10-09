from pygam import LinearGAM
from model.AbstractModel import AbstractModel
from sklearn.model_selection import KFold
import numpy as np
import logging


class GAM(AbstractModel):

    def __init__(self):
        super().__init__()
        self.model = LinearGAM
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
            self.m.gridsearch(X_train, y_train)
            predictions = self.m.predict(X_test)
            rmse_iter, mae_iter, r2_iter = self.score_function(y_test, predictions)
            rmse.append(rmse_iter)
            mae.append(mae_iter)
            r2.append(r2_iter)

        self.m = self.model()
        self.m.gridsearch(x, y)

        return np.mean(rmse), np.mean(mae), np.mean(r2), {}

    def fit(self, x, y, **kwargs):
        pass