import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

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

df = pd.read_csv("result_df3.csv", index_col=[0, 1, 2], header=[0])
df.index = df.index.droplevel(1) # drop run_id column


df2 = pd.read_csv("results_df4_temp.csv", sep=' ')
df2.insert(0, 'id', 0)
df2.set_index(['id', 'timeStep'], inplace = True)
temp_column = df2.pop('meshDensityCounting-PID16')
#df.reset_index(inplace=True)
df2['meshDensityCounting-PID16'] = temp_column

data = df2
new_column_names = []
for i in range(0,len(data.columns)):
    if i == 0:
        new_column_names.append(data.columns[0])
    else:
        new_column_names.append('measurement_area' + str(i))
data.columns = new_column_names
data.fillna(0, inplace=True)

sums_data= data.groupby(level=['id','timeStep']).sum()
sums_data.drop(columns=['faceId'], inplace=True)
data_final=sums_data

x_tsc = TSCDataFrame.from_frame_list([data_final]).astype(np.float64)

dmd = models.dmd(x_tsc)
dmd_values = dmd.predict(x_tsc.initial_states(), time_values=x_tsc.time_values())
x_predicted_dmd = dmd_values
dmd_mae, dmd_mse, dmd_rmse, dmd_mape, dmd_r_squared = errors.compiled_errors(x_tsc, x_predicted_dmd)

num_columns = x_tsc.shape[1]
mae_error = []
mse_error = []
rmse_error = []

for i in range(1, num_columns +1):
    if i == num_columns:
        dmd = models.dmd(x_tsc)
    else:
        dmd = models.dmd(x_tsc, rank = i)
    
    dmd_values = dmd.predict(x_tsc.initial_states(), time_values=x_tsc.time_values())
    x_predicted_dmd = dmd_values
    dmd_mae, dmd_mse, dmd_rmse, dmd_mape, dmd_r_squared = errors.compiled_errors(x_tsc, x_predicted_dmd)

    mae_error.append(dmd_mae)
    mse_error.append(dmd_mse)
    rmse_error.append(dmd_rmse)

plt.figure()
plt.plot(mae_error)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('MAE error')
plt.title('Plot of mae_error')

plt.savefig('plot_mae_multi.png')

time_delay_embed = TSCTakensEmbedding(delays=10, lag=0, frequency=1, kappa=0).fit(x_tsc)
embed_values = time_delay_embed.transform(x_tsc)
embed_values_new = time_delay_embed.inverse_transform(embed_values)
#temp = embed_values.xs(2, level ='ID')

dmd = models.dmd(embed_values)
dmd_values_new = dmd.predict(embed_values.initial_states(), time_values=embed_values.time_values())
x_predicted_dmd_new = dmd_values_new

x_alternate = time_delay_embed.inverse_transform(dmd_values_new)

    #dmd_mae_new, dmd_mse_new, dmd_rmse_new, dmd_mape_new, dmd_r_squared_new = errors.compiled_errors(embed_values, x_predicted_dmd_new)
x_time_delay_init = x_tsc.iloc[10:]
dmd_mae_embed, dmd_mse_embed, dmd_rmse_embed, dmd_mape_embed, dmd_r_squared_embed = errors.compiled_errors(x_time_delay_init, x_alternate)

num_columns = embed_values.shape[1]
mae_error2 = []
mse_error2 = []
rmse_error2 = []
x_time_delay_init = x_tsc.iloc[10:]

for i in range(1, num_columns +1):
    if i == num_columns:
        dmd = models.dmd(embed_values)
    else:
        dmd = models.dmd(embed_values, rank = i)
    
    dmd_values_new = dmd.predict(embed_values.initial_states(), time_values=embed_values.time_values())
    x_predicted_dmd_new = dmd_values_new

    x_alternate = time_delay_embed.inverse_transform(dmd_values_new)

    #dmd_mae_new, dmd_mse_new, dmd_rmse_new, dmd_mape_new, dmd_r_squared_new = errors.compiled_errors(embed_values, x_predicted_dmd_new)
    x_time_delay_init = x_tsc.iloc[10:]
    dmd_mae_embed, dmd_mse_embed, dmd_rmse_embed, dmd_mape_embed, dmd_r_squared_embed = errors.compiled_errors(x_time_delay_init, x_alternate)
    mae_error2.append(dmd_mae_embed)
    mse_error2.append(dmd_mse_embed)
    rmse_error2.append(dmd_rmse_embed)

plt.figure()
plt.plot(mae_error2)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('MAE error')
plt.title('Plot of mae_error_embed')

plt.savefig('plot_mae_embed_multi2.png')
mae_error3 = []
mse_error3 = []
rmse_error3 = []

for delay in range(1,30):
    time_delay_embed = TSCTakensEmbedding(delays=delay, lag=0, frequency=1, kappa=0).fit(x_tsc)
    embed_values = time_delay_embed.transform(x_tsc)
    embed_values_new = time_delay_embed.inverse_transform(embed_values)
    num_columns = embed_values.shape[1]

    x_time_delay_init = x_tsc.iloc[delay:]

    dmd = models.dmd(embed_values)
    dmd_values_new = dmd.predict(embed_values.initial_states(), time_values=embed_values.time_values())
    x_predicted_dmd_new = dmd_values_new

    x_alternate = time_delay_embed.inverse_transform(dmd_values_new)
    dmd_mae_embed, dmd_mse_embed, dmd_rmse_embed, dmd_mape_embed, dmd_r_squared_embed = errors.compiled_errors(x_time_delay_init, x_alternate)
    mae_error3.append(dmd_mae_embed)
    mse_error3.append(dmd_mse_embed)
    rmse_error3.append(dmd_rmse_embed)

plt.figure()
plt.plot(mae_error3)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('MAE error')
plt.title('Plot of mae_error_embed_multi')

plt.savefig('plot_mae_delay_multi2.png')

print ('hello')