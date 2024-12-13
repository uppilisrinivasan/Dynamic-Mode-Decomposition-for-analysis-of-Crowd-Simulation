import matplotlib.pyplot as plt
#import PyQt5
import numpy as np
import pandas as pd
import scipy

from datafold import (
    EDMD,
    DMDStandard,
    GaussianKernel,
    TSCPolynomialFeatures,
    TSCRadialBasis,
    TSCDataFrame
)
from datafold.utils._systems import Hopf
from datafold.utils.general import generate_2d_regular_mesh

import errors
import models
import utils

df = pd.read_csv("result_df2.csv", index_col=[0, 1, 2], header=[0])
#df = pd.read_csv("result_df.csv", index_col=[0, 1, 2], header=[0])
df.index = df.index.droplevel(1) # drop run_id column
#df_reshaped = df.pivot(columns='faceId')  # introduce columns for faceId values

data = df.xs(0, level='id')
#data = df_reshaped[df_reshaped.index.names == 'id']
#data = data.drop(['id','run_id'], axis = 'columns')
#data.to_csv('out.csv', sep='\t')
new_column_names = []
for i in range(0,len(data.columns)):
    new_column_names.append('face' + str(i+1))

data.columns = new_column_names

ped_leaving = utils.compute_ped_leaving(data)

data_with_targets = data.copy()
data_with_targets['ped_leaving'] = ped_leaving
ped_leaving = np.array(ped_leaving).reshape(-1, 1)
x_tsc = TSCDataFrame.from_frame_list([data]).astype(np.float64)  # convert to TSC
targets = TSCDataFrame.from_array(ped_leaving, time_values = x_tsc.time_values() )

x_tsc_new = TSCDataFrame.from_frame_list([data_with_targets]).astype(np.float64)  # convert to TSC

dmd = models.dmd(x_tsc)
dmd_modes = dmd.dmd_modes
dmd_eigenvalues = dmd.eigenvalues_
dmd_dt = dmd.dt_

dmd_values = dmd.predict(x_tsc.initial_states(), time_values=x_tsc.time_values())
x_predicted_dmd = dmd_values

dmd_mae, dmd_mse, dmd_rmse, dmd_mape, dmd_r_squared = errors.compiled_errors(x_tsc, x_predicted_dmd)

#out of sample prediction
dmd_values_oos = dmd.predict(x_tsc.initial_states(), time_values=np.linspace(1,100,125))


edmd_poly = models.edmd_poly(x_tsc, degree=2)
edmd_poly_values = edmd_poly.predict(
    x_tsc.initial_states(), time_values=x_tsc.time_values()
)

x_predicted_edmd_poly = edmd_poly_values

edmd_poly_mae, edmd_poly_mse, edmd_poly_rmse, edmd_poly_mape, edmd_poly_r_squared = errors.compiled_errors(x_tsc, x_predicted_edmd_poly)

edmd_rbf = models.edmd_rbf(x_tsc, epsilon = 0.17)
edmd_rbf_values = edmd_rbf.predict(
    x_tsc.initial_states(), time_values=x_tsc.time_values()
)

x_predicted_edmd_rbf = edmd_rbf_values

edmd_rbf_mae, edmd_rbf_mse, edmd_rbf_rmse, edmd_rbf_mape, edmd_rbf_r_squared = errors.compiled_errors(x_tsc, x_predicted_edmd_rbf)

dmd2 = DMDStandard()
dmd2.fit(X=x_tsc, y=targets)
dmd2_values = dmd2.predict(x_tsc.initial_states(), time_values=x_tsc.time_values())

x_predicted_dmd2 = dmd2_values

dmd2_mae, dmd2_mse, dmd2_rmse, dmd2_mape, dmd2_r_squared = errors.compiled_errors(x_tsc, x_predicted_dmd2)

pred_ped_leaving = utils.compute_ped_leaving(dmd2_values)

pred_ped_leaving = np.array(pred_ped_leaving).reshape(-1, 1)

new_targets = TSCDataFrame.from_array(pred_ped_leaving, time_values = x_tsc.time_values())

targets_mae, targets_mse, targets_rmse, targets_mape, targets_r_squared = errors.compiled_errors(targets, new_targets)

#tar_mae, tar_mse, tar_rmse, tar_mape, tar_r_squared = errors.compiled_errors(num_pedestrians,pred_pedestrians)


edmd_poly2 = models.edmd_poly(x_tsc, degree=2, y= targets)
edmd_poly2_values = edmd_poly2.predict(
    x_tsc.initial_states(), time_values=x_tsc.time_values()
)

edmd_rbf2 = models.edmd_rbf(x_tsc, epsilon = 0.17, y=targets)
edmd_rbf2_values = edmd_rbf2.reconstruct(
    x_tsc
)

edmd_rbf2_mae = errors.mae(targets, edmd_rbf2_values)
edmd_rbf2_mse = errors.mse(targets, edmd_rbf2_values)

dmd3 = models.dmd(x_tsc_new)
dmd3.fit(x_tsc_new)
dmd3_values = dmd3.predict(x_tsc_new.initial_states(), time_values=x_tsc_new.time_values())

temp_pred = dmd3_values
x_tsc_new_predicted = temp_pred.iloc[:, :-1]
x_tsc_new_construct = x_tsc_new.iloc[:, :-1]

dmd3_mae, dmd3_mse, dmd3_rmse, dmd3_mape, dmd3_r_squared = errors.compiled_errors(x_tsc_new_construct, x_tsc_new_predicted)

edmd_poly3 = models.edmd_poly(x_tsc_new, degree = 2)
edmd_poly3_values = edmd_poly3.predict(
    x_tsc_new.initial_states(), time_values=x_tsc_new.time_values()
)
temp_pred_poly3 = edmd_poly3_values
x_tsc_new_predicted_poly3 = temp_pred_poly3.iloc[:, :-1]

edmd_poly3_mae, edmd_poly3_mse, edmd_poly3_rmse, edmd_poly3_mape, edmd_poly3_r_squared = errors.compiled_errors(x_tsc_new_construct, x_tsc_new_predicted_poly3)

sum_pred_ped_edmd_poly3= utils.compute_ped_leaving(x_tsc_new_predicted_poly3)

pred_ped_edmd_poly3 = edmd_poly3_values.iloc[:, -1]

ped_edmd_poly3_mae = errors.mae(pred_ped_edmd_poly3,sum_pred_ped_edmd_poly3)

edmd_rbf3 = models.edmd_rbf(x_tsc_new, epsilon = 0.17)
edmd_rbf3_values = edmd_rbf3.predict(
    x_tsc_new.initial_states(), time_values=x_tsc_new.time_values()
)
temp_pred_rbf3 = edmd_rbf3_values
x_tsc_new_predicted_rbf3 = temp_pred_rbf3.iloc[:, :-1]

edmd_rbf3_mae, edmd_rbf3_mse, edmd_rbf3_rmse, edmd_rbf3_mape, edmd_rbf3_r_squared = errors.compiled_errors(x_tsc_new_construct, x_tsc_new_predicted_rbf3)

sum_pred_ped_edmd_rbf3= utils.compute_ped_leaving(x_tsc_new_predicted_rbf3)

pred_ped_edmd_rbf3 = edmd_rbf3_values.iloc[:, -1]

ped_edmd_rbf3_mae = errors.mae(pred_ped_edmd_rbf3,sum_pred_ped_edmd_rbf3)
print('hello')

# modeling melburnians paper from daniel
