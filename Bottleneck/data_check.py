
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
df_reshaped = df.pivot(columns='faceId')  # introduce columns for faceId values

data = df_reshaped.xs(0, level='id')
data2 = df_reshaped.xs(1, level='id')
data_mae = errors.mae(data,data2)
print('____')