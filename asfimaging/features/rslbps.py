from typing import Any
from asfimaging.shtools import sphericalharmonicsfit as shfit
from asfimaging.shtools import sphericalharmonics as sh
import pandas as pd
import os
import nibabel as nib

import numpy as np
from asfimaging.features.features import Features

class RSLBPS(Features):

    def __init__(self, config, scheme):
        super().__init__(config)
        self.scheme = scheme

    def calculate(self, dwi_data, mask):
        scheme = self.scheme
        sh_order = self.config["real_data_acquisition"]['sh_order']
        LM = sh.compute_lm_ordervector(sh_order)
        unique_zonal_orders = np.unique(LM[:,0])
        N_unique_zonal_orders = len(unique_zonal_orders)

        # Initialize lists to store results
        zos = []
        TEs = []
        bs = []
        dshellsList = []

        dshells = scheme[(scheme.b>0)].dshell.unique()
        n_tevalues = 1
        tevalue = self.config['real_data_acquisition']['TE']
        N_unique_shells = len(dshells)*n_tevalues

        N_zonal_spectra = int(sh_order/2)+1


        nR,nC,nS,nB =dwi_data.shape
        RSLBPS = np.zeros((nR,nC,nS, len(dshells)*n_tevalues*N_unique_zonal_orders))

        # Find the indices of the b=0 data, because we normalize data with respect to that:
        idxs_b0 = scheme[(scheme.b==0) & (scheme.te==tevalue)].index
        b0 = np.nanmean(dwi_data[...,idxs_b0],axis=-1)

        cnt=0
        cnt_mean = 0

        if self.config['real_data_preprocessing']['enforce_non_negativity']: 
             type_ = 'relaxedpos'
        else:
             type_ = 'ls'

        for n_dshell, dshell in enumerate(dshells):
            dirs_x = scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].x.values
            dirs_y = scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].y.values
            dirs_z = scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].z.values
            dirs_xyz = np.vstack([dirs_x,dirs_y,dirs_z]).T

            idxs_selection = scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].index

            shcoeff = shfit.fit_sh_all_data(np.nan_to_num(dwi_data[...,idxs_selection]/b0[...,np.newaxis]),
                                        dirs_xyz,
                                        sh_order,
                                        n_workers = 12,
                                        mask = mask,
                                        type = type_, 
                                        regularization = 'lb',
                                        lambda_reg = 1e-6,
                                        progressevery = 1000,
                                        cvxpyverbose = False, 
                                        solver = 'SCS', 
                                        inputscale = 1)


            print(shcoeff.shape)
            print(RSLBPS.shape)
            for k,zonal_order in enumerate(unique_zonal_orders):
                idx = np.where(LM[:,0]==zonal_order)
                print(zonal_order)
                print(np.sum(shcoeff[...,idx],axis=-1).shape)

                RSLBPS[:,:,:,cnt] = np.reshape(np.sqrt(np.sum(shcoeff[...,idx]**2,axis=-1)),(RSLBPS[:,:,:,cnt]).shape)
                zos.append(zonal_order)
                bs.append(scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].b.unique()[0])
                TEs.append(scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].te.unique()[0])
                dshellsList.append(scheme[(scheme.dshell==dshell) & (scheme.te==tevalue)].dshell.unique()[0])
                cnt += 1


        scheme_sph_dict = {}
        scheme_sph_dict['zonal_order'] = zos
        scheme_sph_dict['te'] = TEs
        scheme_sph_dict['dshell'] = dshellsList
        scheme_sph_dict['b'] = bs

        scheme_sph_df = pd.DataFrame(scheme_sph_dict)

        return RSLBPS, scheme_sph_df


    def save(self, data_affine, RSLBPS, scheme_sph_df):
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'rslbps.nii.gz')
        nib.save(nib.Nifti1Image(RSLBPS, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'rslbps_scheme.pkl')
        scheme_sph_df.to_pickle(filename)
        return None
