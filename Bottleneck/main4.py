import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from datafold import (
    EDMD,
    DMDStandard,
    GaussianKernel,
    TSCPolynomialFeatures,
    TSCRadialBasis,
    TSCDataFrame,
    TSCTakensEmbedding
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
    if i == 0:
        new_column_names.append(data.columns[0])
    else:
        new_column_names.append('measurement_area' + str(i))

data.columns = new_column_names
#data = data.fillna(0)
data_init = data.iloc[:, :-1]
data_init.dropna(axis=0, inplace=True)
targets_temp = data.iloc[:, -1:]

sums_data= data_init.groupby(level='timeStep').sum()
sums_data.drop(columns=['faceId'], inplace=True)
data_final=sums_data #.iloc[:, :-1]
targets_init = targets_temp.groupby(level='timeStep').sum()
#targets_init = sums_data.iloc[:, -1:]
temp_array = targets_init.to_numpy()
temp_array2 = utils.compute_ped_leaving(targets_init)
temp_array= temp_array.reshape(len(temp_array),)

temp_array2= temp_array2.reshape(len(temp_array2),)

# adding a column to of number of pedestrians leaving the room
data_final['ped_leaving'] = temp_array2   
x_tsc = TSCDataFrame.from_frame_list([data_final]).astype(np.float64)  # convert to TSC
real_time1 = utils.compute_time_for_all_to_leave(x_tsc)

num_columns = x_tsc.shape[1]
mae_error = []
mse_error = []
rmse_error = []

for delay in range(25,30):
    time_delay_embed = TSCTakensEmbedding(delays=delay, lag=0, frequency=1, kappa=0).fit(x_tsc)
    embed_values = time_delay_embed.transform(x_tsc)
    embed_values_new = time_delay_embed.inverse_transform(embed_values)

    num_columns = embed_values.shape[1]

    x_time_delay_init = x_tsc.iloc[delay:]

    dmd = models.dmd(embed_values)

    dmd_values_new = dmd.predict(embed_values.initial_states(), time_values=embed_values.time_values())
    x_predicted_dmd_new = dmd_values_new

    x_alternate = time_delay_embed.inverse_transform(dmd_values_new)

    dmd_time2 = utils.compute_time_for_all_to_leave(x_alternate)

    dmd_mae_embed, dmd_mse_embed, dmd_rmse_embed, dmd_mape_embed, dmd_r_squared_embed = errors.compiled_errors(x_time_delay_init, x_alternate)
    dmd_mae_embed2, dmd_mse_embed2, dmd_rmse_embed2, dmd_mape_embed2, dmd_r_squared_embed2 = errors.compiled_errors(x_time_delay_init.iloc[:, :-1], x_alternate.iloc[:, :-1])

    dmd_time2_mae_embed, dmd_time2_mse_embed, dmd_time2_rmse_embed, dmd_time2_mape_embed, dmd_time2_r_squared_embed = errors.compiled_errors(real_time1, dmd_time2)

    mae_error.append(dmd_mae_embed)
    mse_error.append(dmd_mse_embed)
    rmse_error.append(dmd_rmse_embed)

    plt.figure()
plt.plot(mae_error)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('MAE error')
plt.title('Plot of mae_error_embed')

plt.savefig('plot_mae_delay.png')

plt.figure()
plt.plot(mse_error)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('MSE error')
plt.title('Plot of mse_error_embed')

plt.savefig('plot_mse_delay.png')

plt.figure()
plt.plot(rmse_error)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('RMSE error')
plt.title('Plot of rmse_error_embed')

plt.savefig('plot_rmse_delay.png')
# Show the plot
#plt.show()
print('hello')