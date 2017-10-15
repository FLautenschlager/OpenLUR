import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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

model_var = pd.DataFrame(sio.loadmat(paths.rdir + 'model_ha_variables.mat')['model_variables'])
model_var.columns = ["x","y","population","industry","floorlevel","heating","elevation","streetsize","signaldist","streetdist","slope","expo","traffic","streetdist_m","streetdist_l","trafficdist_l","trafficdist_h","traffic_tot"]

# Select test and training dataset for 10 fold cross validation
kf = KFold(n_splits=10, shuffle=True)

rmse_model = []
rsq_model = []
devexpl_model = []
fac2_model = []
rsqval_model = []

# Note: Hasenfratz does the 10 fold cross validation 40 times
# We do it only once for now
for train_index_calib, test_index_calib in kf.split(calib_data):
    # print('TRAIN:', train_index_calib, 'TEST:', test_index_calib)
    train_calib_data = calib_data.iloc[train_index_calib]
    test_calib_data = calib_data.iloc[test_index_calib]

    # Select test data from model_var (data NOT used for calibration)
    # Do this by finding all rows in model_var whose x and y coordinates are not
    # in train_calib_data
    ind_keys = ['x', 'y']
    ind_train_calib = train_calib_data.set_index(ind_keys).index
    ind_test_calib = test_calib_data.set_index(ind_keys).index
    ind_model_var = model_var.set_index(ind_keys).index

    test_data = model_var[~ind_model_var.isin(ind_train_calib)]


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
    model = mgcv.gam(formula, family, data=train_calib_data)
    su = base.summary(model)

    # Predict the test data
    pred_data = stats.predict(model, newdata=test_data, type='response')
    test_data_predictions = test_data.assign(prediction=pred_data)
    # Check how large the error is with the remaining 10% of data
    error_model = test_calib_data['pm_measurement']-test_data_predictions['prediction']
    # Drop all NaN's
    error_model = error_model.dropna()
    print(error_model)
    # Calculate Root-mean-square error model
    rmse_model.append(np.sqrt(np.mean(error_model**2)))
    # rsq_model <- c(rsq_model, su$r.sq)
    # devexpl_model <- c(devexpl_model, su$dev.expl*100)
    # fac2_ind <- which(calib_data[-ind,3]/pred_data<=2 & calib_data[-ind,3]/pred_data>=0.5)
    # fac2_model <- c(fac2_model, length(fac2_ind)/length(ind_var)*100)
    # # calculate R2 between predicted and measured concentration
    # lt1 <- lm(calib_data[-ind,3]~pred_data)
    # rsqval_model <- c(rsqval_model, summary(lt1)$r.squared)

    # print(assign_test)
    # print(pred_data)
    # print(len(pred_data))
    # print(test_calib_data)
    # print(rmse_model)
    # print(su)
    # input()


print('Root-mean-square error:', np.mean(rmse_model))
