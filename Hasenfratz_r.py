import scipy.io as sio
import pandas as pd
import paths
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, pandas2ri
from rpy2.robjects.packages import importr

# Load data
pm_ha = sio.loadmat(paths.extdatadir + 'pm_ha_ext_01042012_30062012.mat')['pm_ha']

# mgcv is the R package with the GAM implementation
mgcv = importr('mgcv')
base = importr('base')
stats = importr('stats')

# Activate implicit conversion of pandas to rpy2 and vice versa
pandas2ri.activate()

# Prepare data
data_1 = pd.DataFrame(pm_ha[:,:3])
data_2 = pd.DataFrame(pm_ha[:,7:])
calib_data = pd.concat([data_1, data_2], axis=1)
calib_data.columns = ["x","y","pm_measurement","population","industry","floorlevel","heating","elevation","streetsize","signaldist","streetdist","slope","expo","traffic","streetdist_m","streetdist_l","trafficdist_l","trafficdist_h","traffic_tot"]

# This is the formula for the GAM
# From https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.terms.html:
# "s()" defines a smooth term in R
# "bs" is the basis of the used smooth class
# "cr" declares a cubic spline basis
# "k" defines the dimension of the basis (upper limit on degrees of freedom)
formula = robjects.Formula('pm_measurement~s(industry,bs="cr",k=3)+s(floorlevel,bs="cr",k=3)+s(elevation,bs="cr",k=3)+s(slope,bs="cr",k=3)+s(expo,bs="cr",k=3)+streetsize+s(traffic_tot,bs="cr",k=3)+s(streetdist_l,bs="cr",k=3)')
# Hasenfratz uses a Gamma distribution with a logarithmic link
family = stats.Gamma(link='log')

# Train model
model = mgcv.gam(formula, family, data=calib_data)
su = base.summary(model)

# model_var = pd.DataFrame(sio.loadmat(paths.rdir + 'model_ha_variables.mat')['model_variables'])
# model_var.columns = ["x","y","population","industry","floorlevel","heating","elevation","streetsize","signaldist","streetdist","slope","expo","traffic","streetdist_m","streetdist_l","trafficdist_l","trafficdist_h","traffic_tot"]
#
# pred_data = stats.predict(model, newdata=model_var, type='response')

print(su)
