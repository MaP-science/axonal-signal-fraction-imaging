import numpy as np
import pandas as pd
import os
from asfimaging.features.features import Features
import nibabel as nib


class Means(Features):

    def __init__(self, config, scheme):
        super().__init__(config)
        self.scheme = scheme

    def calculate(self, data, mask):
        # Get only rslbps for high b-values
        data_DWI_img = data
        scheme_df = self.scheme
        tevalue = self.config['real_data_acquisition']['TE']
        b0 = np.mean(data_DWI_img[...,scheme_df[(scheme_df.b==0)&(scheme_df.te==tevalue)].index], axis=-1)
        df_b_mask = (scheme_df.b==self.config['real_data_acquisition']['high_bmin']) | (scheme_df.b==self.config['real_data_acquisition']['high_bmax']) 
        scheme_df_highb = scheme_df[df_b_mask]
        data_highb = data_DWI_img[...,scheme_df_highb.index]

        # Initialize variables
        nR,nC,nS,n4 = data_highb.shape
        num_means = int(scheme_df_highb.b.unique().shape[0])
        means = np.zeros((nR,nC,nS,num_means))


        TEs1 = []
        bs1 = []
        dshellsList1 = []


        for j, bval in enumerate(scheme_df_highb.b.unique()):
            signal = data_highb[...,((scheme_df_highb.b==bval) & (scheme_df_highb.te==tevalue))] 
            means[...,j] = np.mean(signal/b0[...,np.newaxis], axis=-1)

            bs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].b.unique()[0])
            TEs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].te.unique()[0])
            dshellsList1.append(scheme_df_highb[(scheme_df_highb.b==bval)].dshell.unique()[0])


        scheme_dict = {}
        scheme_dict['te1'] = TEs1
        scheme_dict['dshell1'] = dshellsList1
        scheme_dict['b1'] = bs1

        scheme_means= pd.DataFrame(scheme_dict)

        return np.nan_to_num(means), scheme_means

    def save(self, data_affine, means, scheme_means):
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'means.nii.gz')
        nib.save(nib.Nifti1Image(means, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'means_scheme.pkl')
        scheme_means.to_pickle(filename)
        return None
