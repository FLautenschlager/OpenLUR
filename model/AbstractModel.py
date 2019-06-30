from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import logging

class AbstractModel:

    def __init__(self):
        self.m = None
        self.model = None
        self.log = logging.getLogger(self.__class__.__name__)

    def score_function(self, target, prediction):
        rmse = np.sqrt(mean_squared_error(target, prediction))
        mae = mean_absolute_error(target, prediction)
        r2 = r2_score(target, prediction)
        return rmse, mae, r2

    def fit(self, x, y, modeldict=None):
        if self.m is None:
            if modeldict:
                self.m = self.model(**modeldict)
            else:
                self.m = self.model()
        self.m.fit(x, y)

    def param_search(self, x, y, score_funtion=None, score_rank_lowest=False, processes=1, **kwargs):
        print("Implement the param_search function!")

    def predict(self, x):
        if self.m:
            return self.m.predict(x)
        else:
            raise NotImplementedError("Model has not been trained yet.")

    @staticmethod
    def concat_results(rmse, mae, r2, model_dict={}):
        result = pd.Series()
        result['rmse'] = rmse
        result['mae'] = mae
        result['r2'] = r2
        result["model_dict"] = model_dict

        return result

