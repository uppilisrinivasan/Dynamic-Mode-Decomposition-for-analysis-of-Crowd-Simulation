import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
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


def dmd(x_tsc: TSCDataFrame, rank=None, y=None):
    if rank is not None:
        assert isinstance(rank, int), "Variable 'rank' is not an integer."
        dmd = DMDStandard(rank= rank)
        if y is not None:
            dmd.fit(x_tsc,y=y)
        else:
            dmd.fit(x_tsc)
    else:
        dmd = DMDStandard()
        if y is not None:
            dmd.fit(x_tsc,y=y)
        else:
            dmd.fit(x_tsc)
    return dmd

def edmd_poly(x_tsc: TSCDataFrame, degree: int, y=None):
    dict_step = [
    (
        "polynomial",
        TSCPolynomialFeatures(degree= degree),
    )
    ]
    if y is not None:
        edmd_poly = EDMD(dict_steps=dict_step,dmd_model=DMDStandard(), include_id_state=True).fit(X=x_tsc, y=y)
    else:
        edmd_poly = EDMD(dict_steps=dict_step,dmd_model=DMDStandard(), include_id_state=True).fit(X=x_tsc)
    
    return edmd_poly

def edmd_rbf(x_tsc: TSCDataFrame, epsilon: float,y=None):
    dict_step = [
    (
        "rbf",
        TSCRadialBasis(
            kernel=GaussianKernel(epsilon=epsilon)#, center_type="initial_condition" #change center_type
        ),
    )
    ]
    if y is not None:
        edmd_rbf = EDMD(dict_steps=dict_step, include_id_state=True).fit(X=x_tsc, y=y)
    else:
        edmd_rbf = EDMD(dict_steps=dict_step, include_id_state=True).fit(X=x_tsc)
    
    return edmd_rbf
