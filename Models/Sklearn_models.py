from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from Models.SklearnWrapper import SklearnWrapper
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from gwr.sel_bw import Sel_BW
from gwr.gwr import GWR as gwr

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class RF(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.model = RandomForestRegressor
		self.name = name


class RF100(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.name = name

	def model(self):
		return RandomForestRegressor(n_estimators=100)


class Adaboost(SklearnWrapper):
	def __init__(self, njobs, niter=40, verbosity=0):
		super().__init__(njobs, name, niter=niter, verbosity=verbosity)
		self.model = AdaBoostRegressor
		self.name = name

class SGD(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.model = SGDRegressor
		self.name = name


class MLP(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.model = MLPRegressor
		self.name = name


class KNN(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.model = KNeighborsRegressor
		self.name = name


class GWR(SklearnWrapper):
	def __init__(self, njobs, name, niter=40, verbosity=0):
		super().__init__(njobs, niter=niter, verbosity=verbosity)
		self.name = name

	def calculateModel(self, inputs):
		train_data, test_data, target, columns = inputs

		X_train = train_data[columns].as_matrix()
		X_test = test_data[columns].as_matrix()

		y_train = train_data[target].as_matrix().reshape((-1,1))
		y_test = test_data[target].as_matrix().reshape((-1,1))

		loc_train = train_data[['x', 'y']].as_matrix()
		loc_test = test_data[['x', 'y']].as_matrix()

		bw = Sel_BW(loc_train, y_train, X_train).search(criterion='AICc')
		model = gwr(loc_train, y_train, X_train, bw=bw)
		model.fit()

		pred = model.predict(loc_test, X_test).predy

		rmse = np.sqrt(mean_squared_error(y_test, pred))
		r2 = r2_score(y_test, pred)

		self.print('Root-mean-square error: {} particles/cm^3'.format(rmse), 2)
		self.print('R2: {}'.format(r2), 2)

		return rmse, r2



