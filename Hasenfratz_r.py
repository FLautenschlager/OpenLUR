import scipy.io as sio
import paths
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

ext_data = sio.loadmat(paths.extdatadir + 'pm_ha_ext_01042012_30062012.mat')['pm_ha']

mgcv = importr('mgcv')
base = importr('base')
stats = importr('stats')

robjects.globalenv['industry'] = FloatVector(ext_data[:,8])
robjects.globalenv['floorlevel'] = FloatVector(ext_data[:,9])
robjects.globalenv['elevation'] = FloatVector(ext_data[:,11])

robjects.globalenv['pm_measurement'] = FloatVector(ext_data[:,2])

formula = robjects.Formula('pm_measurement~s(industry,bs="cr",k=3)+s(floorlevel,bs="cr",k=3)+s(elevation,bs="cr",k=3)')
family = stats.Gamma(link='log')

model = mgcv.gam(formula, family)

su = base.summary(model)

print(su)
