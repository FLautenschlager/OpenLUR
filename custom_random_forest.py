import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class RandomForest(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.estimator_increment = 10
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        n_iter = 2
        self.iterative_fit(X, y, n_iter=n_iter, refit=True)
        while not self.configuration_fully_fitted():
            n_iter *= 2
            self.iterative_fit(X, y, n_iter=n_iter)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.ensemble import RandomForestRegressor

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if self.max_depth == "None" or self.max_depth is None:
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.max_features = float(self.max_features)
            if self.bootstrap == "True":
                self.bootstrap = True
            else:
                self.bootstrap = False
            if self.max_leaf_nodes == "None" or self.max_leaf_nodes is None:
                self.max_leaf_nodes = None
            self.min_impurity_decrease = float(self.min_impurity_decrease)

            self.estimator = RandomForestRegressor(
                n_estimators=0,
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                warm_start=True)
        self.estimator.n_estimators += n_iter
        self.estimator.n_estimators = min(self.estimator.n_estimators,
                                          self.n_estimators)

        self.estimator.fit(X, y)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False

        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RF',
                'name': 'Random Forest Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'prefers_data_normalized': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 1, 1000, default_value=2)
        criterion = Constant("criterion", 'mse')
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1.0, default_value=1.0)
        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = \
            UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            'min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")

        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease, bootstrap])

        return cs
