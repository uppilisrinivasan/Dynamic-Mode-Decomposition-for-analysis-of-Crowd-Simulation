import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import math
from sklearn import preprocessing

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
from datafold.dynfold.dmd import DMDControl

def compute_ped_leaving(df: pd.DataFrame):
    num_pedestrians = df.to_numpy().flatten()
    ped_leaving = []
    ped_leaving.append(0)
    count_pedestrians = num_pedestrians[0]
    for i in range(1,len(num_pedestrians)):
        if num_pedestrians[i] > count_pedestrians:
            count_pedestrians = num_pedestrians[i]
            ped_leaving.append(ped_leaving[i-1])
        else:
            ped_leaving.append(count_pedestrians - num_pedestrians[i])
    
    ped_leaving_final = np.array(ped_leaving)
    return ped_leaving_final

def compute_time_for_all_to_leave(df: TSCDataFrame):
    ped_leaving = df.iloc[:, -1:].to_numpy().flatten()
    lower_limit = int(max(ped_leaving))
    upper_limit = math.ceil(max(ped_leaving))

    temp = np.where((ped_leaving >= lower_limit) & (ped_leaving <= upper_limit))[0]
    
    timestep = temp[0]
    return timestep