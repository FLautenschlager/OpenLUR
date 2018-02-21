from Models.SklearnWrapper import SklearnWrapper
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

def getRF(njobs, niter=40, verbosity=0):
	return SklearnWrapper(njobs, RandomForestRegressor, niter, verbosity)

def getSGD(njobs, niter=40, verbosity=0):
	return SklearnWrapper(njobs, SGDRegressor, niter, verbosity)