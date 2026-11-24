import numpy as np
import pandas as pd
from asfimaging.features.features import Features, safe_divide
import nibabel as nib


class Means(Features):


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
        num_means = len(np.sort(scheme_df_highb.b.unique()))
        means = np.zeros((nR,nC,nS,num_means))


        TEs1 = []
        bs1 = []
        dshellsList1 = []


        for j, bval in enumerate(np.sort(scheme_df_highb.b.unique())):  # [b1, b2]
            signal = data_highb[...,((scheme_df_highb.b==bval) & (scheme_df_highb.te==tevalue))] 
            means[...,j] = np.mean(safe_divide(signal, b0[...,np.newaxis], mask, label='Means b0 normalisation'), axis=-1)

            bs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].b.unique()[0])
            TEs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].te.unique()[0])
            dshellsList1.append(scheme_df_highb[(scheme_df_highb.b==bval)].dshell.unique()[0])


        scheme_dict = {}
        scheme_dict['te'] = TEs1
        scheme_dict['dshell'] = dshellsList1
        scheme_dict['b'] = bs1

        scheme_means= pd.DataFrame(scheme_dict)

        return means, scheme_means

    def save(self, data_affine, means, scheme_means):
        directory = self.output_directory()
        nib.save(nib.Nifti1Image(means, data_affine), directory / 'means.nii.gz')
        scheme_means.to_pickle(directory / 'means_scheme.pkl')
        return None
