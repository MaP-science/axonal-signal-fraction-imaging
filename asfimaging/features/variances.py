import numpy as np
import pandas as pd
import nibabel as nib
from asfimaging.features.features import Features
import os

class Variances(Features):

    def __init__(self, config, scheme):
        super().__init__(config)
        self.scheme = scheme

    def calculate(self, dwi_data, mask):
        scheme_df = self.scheme
        tevalue = self.config['real_data_acquisition']['TE']
        # Get only rslbps for high b-values
        b0 = np.mean(dwi_data[...,scheme_df[(scheme_df.b==0)&(scheme_df.te==tevalue)].index], axis=-1)
        df_b_mask = (scheme_df.b==self.config['real_data_acquisition']['high_bmin']) | (scheme_df.b==self.config['real_data_acquisition']['high_bmax']) 
        scheme_df_highb = scheme_df[df_b_mask]
        data_highb = dwi_data[...,scheme_df_highb.index]

        # Initialize variables
        nR,nC,nS,n4 = data_highb.shape
        num_variances = int(scheme_df_highb.b.unique().shape[0])
        variances = np.zeros((nR,nC,nS,num_variances))


        TEs1 = []
        bs1 = []
        dshellsList1 = []

        for j, bval in enumerate(scheme_df_highb.b.unique()):
            signal = data_highb[...,((scheme_df_highb.b==bval) & (scheme_df_highb.te==tevalue))] 
            variances[...,j] = np.var(signal/b0[...,np.newaxis], axis=-1) 

            bs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].b.unique()[0])
            TEs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].te.unique()[0])
            dshellsList1.append(scheme_df_highb[(scheme_df_highb.b==bval)].dshell.unique()[0])


        scheme_dict = {}
        scheme_dict['te1'] = TEs1
        scheme_dict['dshell1'] = dshellsList1
        scheme_dict['b1'] = bs1

        scheme_variances = pd.DataFrame(scheme_dict)

        return np.nan_to_num(variances), scheme_variances

    def save(self, data_affine, variances, scheme_variances):
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'variances.nii.gz')
        nib.save(nib.Nifti1Image(variances, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'variances_scheme.pkl')
        scheme_variances.to_pickle(filename)
        return None