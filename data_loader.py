import numpy as np
import pandas as pd

def load_data(num_data=int):
    ct_X, ct_y, num_classes = load_ct_data()
    tau_X, tau_y, _ = load_tau_data()
    fdg_X, fdg_y, _ = load_fdg_data()
    amy_X, amy_y, _ = load_amy_data()

    if num_data == 2:
        return np.concatenate((ct_X,tau_X),axis=0), np.concatenate((ct_y,tau_y),axis=0), num_classes
    elif num_data == 3:
        return np.concatenate((ct_X,tau_X,fdg_X),axis=0), np.concatenate((ct_y,tau_y,fdg_y),axis=0), num_classes
    elif num_data == 4:
        return np.concatenate((ct_X,tau_X,fdg_X,amy_X),axis=0), np.concatenate((ct_y,tau_y,fdg_y,amy_y),axis=0), num_classes
    else:
        raise ValueError