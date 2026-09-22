import numpy as np
import pandas as pd
import nibabel as nib
import os
from asfimaging.features.features import Features



class ZonalRatios(Features):

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
        num_ratios = int((sh_order/2)+1)
        
        ratios = np.zeros((nR,nC,nS,num_ratios))
        
        count=0

        zos = []
        TEs = []
        bs = []
        dshellsList = []

        for k, zo in enumerate(scheme_df.zonal_order.unique()):
            if zo>=0:
                for j, bval in enumerate(scheme_df.b.unique()):
                    if bval>np.min(scheme_df.b.unique()):
                        signal_forward = data_highb_RSLBPS[...,(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])] 
                        signal_backward = data_highb_RSLBPS[...,(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j-1])]

                        ratios[...,count] = np.reshape(np.abs(signal_forward)/np.abs(signal_backward),(ratios[...,count]).shape)   
                        count=count+1

                        zos.append(zo)
                        bs.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].b.unique()[0])
                        TEs.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].te.unique()[0])
                        dshellsList.append(scheme_df[(scheme_df.zonal_order==zo)&(scheme_df.b==scheme_df.b.unique()[j])].dshell.unique()[0])
            

        scheme_ratios_dict = {}
        scheme_ratios_dict['zonal_order'] = zos
        scheme_ratios_dict['te1'] = TEs
        scheme_ratios_dict['dshell1'] = dshellsList
        scheme_ratios_dict['b1'] = bs

        scheme_ratios= pd.DataFrame(scheme_ratios_dict)

        return np.nan_to_num(ratios[...,:count]), scheme_ratios


    def save(self, data_affine, zonal_ratios, scheme):
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'zonal_ratios.nii.gz')
        nib.save(nib.Nifti1Image(zonal_ratios, data_affine), filename)
        filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'zonal_ratios_scheme.pkl')
        scheme.to_pickle(filename)
        return None