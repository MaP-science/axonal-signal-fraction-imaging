from asfimaging.shtools import sphericalharmonicsfit as shfit
from asfimaging.shtools import sphericalharmonics as sh
from asfimaging.config import check_sh_order, n_jobs, sh_fit_settings
import pandas as pd
import nibabel as nib

import numpy as np
from asfimaging.features.features import Features, safe_divide

class RSLBPS(Features):


    def high_b_values(self):
        """The two configured high shells, ascending: [b1, b2]."""
        acquisition = self.config['real_data_acquisition']
        return np.sort([acquisition['high_bmin'], acquisition['high_bmax']])

    def calculate(self, dwi_data, mask):
        scheme = self.scheme
        sh_order = check_sh_order(self.config)
        LM = sh.compute_lm_ordervector(sh_order)
        unique_zonal_orders = np.unique(LM[:,0])
        N_unique_zonal_orders = len(unique_zonal_orders)

        # Initialize lists to store results
        zos = []
        TEs = []
        bs = []
        dshellsList = []

        # Only the two high shells (b1 < b2) are fitted and saved.
        tevalue = self.config['real_data_acquisition']['TE']
        bvalues = self.high_b_values()

        nR,nC,nS,nB =dwi_data.shape
        RSLBPS = np.zeros((nR,nC,nS, len(bvalues)*N_unique_zonal_orders))

        # Find the indices of the b=0 data, because we normalize data with respect to that:
        idxs_b0 = scheme[(scheme.b==0) & (scheme.te==tevalue)].index
        b0 = np.nanmean(dwi_data[...,idxs_b0],axis=-1)

        cnt=0

        fit = sh_fit_settings(self.config)
        type_ = 'relaxedpos' if fit['positivity'] else 'ls'

        for bvalue in bvalues:
            selection = scheme[(scheme.b==bvalue) & (scheme.te==tevalue)]
            dirs_xyz = selection[['x', 'y', 'z']].to_numpy()

            normalized = safe_divide(dwi_data[...,selection.index], b0[...,np.newaxis], mask,
                                     label=f'b0 normalisation (b={bvalue:g})')
            shcoeff = shfit.fit_sh_all_data(normalized,
                                        dirs_xyz,
                                        sh_order,
                                        n_workers = n_jobs(self.config, 'real_data_preprocessing'),
                                        mask = mask,
                                        type = type_,
                                        regularization = 'lb',
                                        lambda_reg = fit['lambda_reg'],
                                        cvxpyverbose = False,
                                        solver = 'SCS',
                                        inputscale = 1)

            for zonal_order in unique_zonal_orders:
                idx = np.where(LM[:,0]==zonal_order)[0]
                RSLBPS[:,:,:,cnt] = np.sqrt(np.sum(shcoeff[...,idx]**2,axis=-1))
                zos.append(zonal_order)
                bs.append(bvalue)
                TEs.append(tevalue)
                dshellsList.append(selection.dshell.iloc[0])
                cnt += 1


        scheme_sph_dict = {}
        scheme_sph_dict['zonal_order'] = zos
        scheme_sph_dict['te'] = TEs
        scheme_sph_dict['dshell'] = dshellsList
        scheme_sph_dict['b'] = bs

        scheme_sph_df = pd.DataFrame(scheme_sph_dict)

        return RSLBPS, scheme_sph_df


    def save(self, data_affine, RSLBPS, scheme_sph_df):
        directory = self.output_directory()
        nib.save(nib.Nifti1Image(RSLBPS, data_affine), directory / 'rslbps.nii.gz')
        scheme_sph_df.to_pickle(directory / 'rslbps_scheme.pkl')
        return None
