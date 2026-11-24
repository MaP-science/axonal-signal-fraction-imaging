import numpy as np
import pandas as pd
import nibabel as nib
from asfimaging.features.features import Features, safe_divide

class Variances(Features):


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
        num_variances = len(np.sort(scheme_df_highb.b.unique()))
        variances = np.zeros((nR,nC,nS,num_variances))


        TEs1 = []
        bs1 = []
        dshellsList1 = []

        for j, bval in enumerate(np.sort(scheme_df_highb.b.unique())):  # [b1, b2]
            signal = data_highb[...,((scheme_df_highb.b==bval) & (scheme_df_highb.te==tevalue))] 
            variances[...,j] = np.var(safe_divide(signal, b0[...,np.newaxis], mask, label='Variances b0 normalisation'), axis=-1)

            bs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].b.unique()[0])
            TEs1.append(scheme_df_highb[(scheme_df_highb.b==bval)].te.unique()[0])
            dshellsList1.append(scheme_df_highb[(scheme_df_highb.b==bval)].dshell.unique()[0])


        scheme_dict = {}
        scheme_dict['te'] = TEs1
        scheme_dict['dshell'] = dshellsList1
        scheme_dict['b'] = bs1

        scheme_variances = pd.DataFrame(scheme_dict)

        return variances, scheme_variances

    def save(self, data_affine, variances, scheme_variances):
        directory = self.output_directory()
        nib.save(nib.Nifti1Image(variances, data_affine), directory / 'variances.nii.gz')
        scheme_variances.to_pickle(directory / 'variances_scheme.pkl')
        return None
