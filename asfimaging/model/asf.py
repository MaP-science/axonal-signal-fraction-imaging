import numpy as np
import scipy.special as ssp
import os
import nibabel as nib
import pandas as pd
import json


class calculate_ASF:
    def __init__(self, config):
        self.config = config

    def load_scheme(self):
        scheme = pd.read_pickle(self.config["data_paths"]["scheme"])
        return scheme

    def signal_spherical_mean(self, b, d_par, d_perp):
        m = np.exp(-b*d_perp) *np.sqrt(np.pi)/2 *(ssp.erf(np.sqrt(b*(d_par-d_perp))))/np.sqrt(b*(d_par-d_perp))
        return m

    def load_synthetic_data(self):
        data = np.load(os.path.join(self.data_path, "data.npy"))
        scheme = self.load_scheme()
        idxs_b5000 = scheme.b==self.config["real_data_acquisition"]["high_bmin"]
        data_b5000_mean = np.mean(data[...,idxs_b5000],axis=-1)
        return data_b5000_mean


    def load_real_data(self, data):
        scheme = self.load_scheme()
        idxs_b0 = scheme.b==0
        data_b0_mean = np.mean(data[...,idxs_b0],axis=-1)

        idxs_b5000 = scheme.b==self.config["real_data_acquisition"]["high_bmin"]
        data_b5000_mean = np.mean(data[...,idxs_b5000],axis=-1)

        data_b5000_normalized = data_b5000_mean/data_b0_mean
        return data_b5000_normalized

    def synthetic_calculate_asf(self, labels):
        d_par = labels[:,0]
        d_perp = labels[:,1]

        sphmean = self.load_synthetic_data()
        C = self.signal_spherical_mean(self.config["real_data_acquisition"]["high_bmin"],d_par,d_perp)
        asf = np.nan_to_num(sphmean/C)
        asf = np.clip(asf, 0, 1)
        return asf

    def real_calculate_asf(self, dpar, dperp, data):

        sphmean = self.load_real_data(data)
        C = self.signal_spherical_mean(self.config["real_data_acquisition"]["high_bmin"],dpar,dperp)
        asf = np.nan_to_num(sphmean/C)
        asf = np.clip(asf, 0, 1)
        return asf
        