import numpy as np
import pandas as pd
import nibabel as nib
from asfimaging.features.features import Features
import os


class Spectra(Features):

    def __init__(self, config, scheme):
        super().__init__(config)
        self.scheme = scheme

    def calculate(self, data_RSLBPS):
        # Get only rslbps for high b-values
        scheme_df = self.scheme
        df_b_mask = (scheme_df.b==self.config['real_data_acquisition']['high_bmin']) | (scheme_df.b==self.config['real_data_acquisition']['high_bmax']) 
        scheme_df = scheme_df[df_b_mask]
        data_highb_RSLBPS = data_RSLBPS[...,scheme_df.index]

        # Initialize variables
        nR,nC,nS,n4 = data_highb_RSLBPS.shape
        sh_order = np.max(scheme_df.zonal_order.values)
        num_spectra = int((sh_order/2)+1)
        spectra_b2 = np.zeros((nR,nC,nS,num_spectra))
        spectra_b1 = np.zeros((nR,nC,nS,num_spectra))

        count=0

        zos_b2 = []
        TEs_b2 = []
        bs_b2 = []
        dshellsList_b2 = []

        zos_b1 = []
        TEs_b1 = []
        bs_b1 = []
        dshellsList_b1 = []

        for k, zo in enumerate(scheme_df.zonal_order.unique()):
            if zo>=0:
                for j, bval in enumerate(scheme_df.b.unique()):
                    if bval>np.min(scheme_df.b.unique()):
                        signal_b2 = data_highb_RSLBPS[...,(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j-1])]
                        signal_b1 = data_highb_RSLBPS[...,(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])]

                        spectra_b2[...,count] = np.reshape(np.abs(signal_b2),(spectra_b2[...,count]).shape) 
                        spectra_b1[...,count] = np.reshape(np.abs(signal_b1),(spectra_b1[...,count]).shape)   
                        count=count+1

                        zos_b2.append(zo)
                        bs_b2.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j-1])].b.unique()[0])
                        TEs_b2.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j-1])].te.unique()[0])
                        dshellsList_b2.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j-1])].dshell.unique()[0])

                        zos_b1.append(zo)
                        bs_b1.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].b.unique()[0])
                        TEs_b1.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].te.unique()[0])
                        dshellsList_b1.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].dshell.unique()[0])
            

        scheme_ratios_dict_b2 = {}
        scheme_ratios_dict_b2['zonal_order'] = zos_b2
        scheme_ratios_dict_b2['te1'] = TEs_b2
        scheme_ratios_dict_b2['dshell1'] = dshellsList_b2
        scheme_ratios_dict_b2['b1'] = bs_b2

        scheme_ratios_dict_b1 = {}
        scheme_ratios_dict_b1['zonal_order'] = zos_b1
        scheme_ratios_dict_b1['te1'] = TEs_b1
        scheme_ratios_dict_b1['dshell1'] = dshellsList_b1
        scheme_ratios_dict_b1['b1'] = bs_b1

        scheme_ratios_b2 = pd.DataFrame(scheme_ratios_dict_b2)
        scheme_ratios_b1 = pd.DataFrame(scheme_ratios_dict_b1)

        return np.nan_to_num(spectra_b2[...,:count]), scheme_ratios_b2, np.nan_to_num(spectra_b1[...,:count]), scheme_ratios_b1

    def save(self, data_affine, spectra_b2, scheme_ratios_b2, spectra_b1, scheme_ratios_b1):
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b2.nii.gz')
        nib.save(nib.Nifti1Image(spectra_b2, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b2_scheme.pkl')
        scheme_ratios_b2.to_pickle(filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b1.nii.gz')
        nib.save(nib.Nifti1Image(spectra_b1, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b1_scheme.pkl')
        scheme_ratios_b1.to_pickle(filename)
        return None