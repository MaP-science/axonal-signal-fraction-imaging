import os
from os import path
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.special as ssp
from pathlib import Path

from asfimaging.shtools import sphericalharmonics as sh
from asfimaging.shtools import sphericalharmonicsfit as shfit
from asfimaging.sh_descoteaux import zeppelin_signal as zeppelin


class SyntheticSignalGenerator:

    def __init__(self,config): 
        self.config = config
        self.sh_order = self.config['real_data_acquisition']['sh_order']
        self.te = self.config['real_data_acquisition']['TE']


    def get_odfs(self):
        """
        Sort the ODFs into bins based on their spherical variance. 
        This is used to sample ODFs with different shapes for the synthetic signal generation.
        """
        # Select ODFs where there actually is a signal
        odf_sh_coeff = np.load(self.config['data_paths']['ODF'])
        odf_sh_coeff = odf_sh_coeff[odf_sh_coeff[...,0]>0]
        odf_sh_coeff = odf_sh_coeff[...,0:len(sh.compute_lm_ordervector(12))]
        odf_sh_coeff = (odf_sh_coeff[:,:]/odf_sh_coeff[:,0:1])/(4*np.pi)**0.5
        sphvarodf = np.sum(odf_sh_coeff[...,1:]**2,axis=-1)/(4*np.pi)
        
        idxs = (sphvarodf>np.percentile(sphvarodf,1)) & (sphvarodf<np.percentile(sphvarodf,99.0))
        odf_sh_coeff = odf_sh_coeff[idxs,:]
        sphvarodf = sphvarodf[idxs]
        
        # Additional histogram and bins
        ha,hb = np.histogram(sphvarodf,bins=10)
        hb = hb[1:]

        list_of_indices_odf = []
        old_lim=0
        for thelim in hb:
            idxsodf = np.where((sphvarodf>=old_lim) & (sphvarodf<thelim))
            list_of_indices_odf.append(idxsodf[0])
            old_lim=thelim


        return odf_sh_coeff, list_of_indices_odf

    def get_random_odf(self, odf_sh_coeff, list_idxs_sh_coeffs): 
        # Randomly select an ODF from the list of ODFs based on their spherical variance bins
        randomint = np.random.randint(0, high=len(list_idxs_sh_coeffs), size=None, dtype=int)
        randomodf = np.random.randint(0, high=len(list_idxs_sh_coeffs[randomint]), size=None, dtype=int)
        
        sh_coeffs_idx = list_idxs_sh_coeffs[randomint][randomodf]
        theodfcoeffs = odf_sh_coeff[sh_coeffs_idx,:]
        return theodfcoeffs


    def coefficient_count_to_order(self, n_coefficients: int) -> int:
        for order in range(0, 13, 2):
            if (order + 1) * (order + 2) // 2 == n_coefficients:
                return order
        raise ValueError(
            f"{n_coefficients} is not an even SH coefficient count through order 12."
        )


    def get_random_parameters(self):
        # Set the fractions of WM (intra, extra), GM and CSF 
        vf_intra = np.random.uniform(low=0.01, high=0.99)
        vf_extra, vf_gm, vf_csf  = np.random.dirichlet(self.config['synthetic_data_parameters']['alphas_dirichlet'],size=1)[0].transpose()
        not_intra = 1 - vf_intra
        
        # Set the volume fractions to amount not occupied by intra
        vf_extra = vf_extra*not_intra
        vf_gm = vf_gm*not_intra
        vf_csf = vf_csf*not_intra

        partial_volume_0 = vf_intra + vf_extra
        relative_vf_intra = vf_intra/partial_volume_0
        relative_vf_extra = vf_extra/partial_volume_0

        # Set the low and high snr uniformly 
        low_snr = self.config['synthetic_data_parameters']['snr'][0]
        high_snr = self.config['synthetic_data_parameters']['snr'][1]
        
        if len(self.config['synthetic_data_parameters']['snr']) > 5:  # If we have a list of SNRs
            # Determine the low and high SNR based on the volume fractions
            if partial_volume_0 > vf_gm + vf_csf:
                low_snr = self.config['synthetic_data_parameters']['snr'][0]
                high_snr = self.config['synthetic_data_parameters']['snr'][1]
            elif vf_gm > partial_volume_0 + vf_csf:
                low_snr = self.config['synthetic_data_parameters']['snr'][2]
                high_snr = self.config['synthetic_data_parameters']['snr'][3]
            elif vf_csf > partial_volume_0 + vf_gm:
                low_snr = self.config['synthetic_data_parameters']['snr'][4]
                high_snr = self.config['synthetic_data_parameters']['snr'][5]
            
        
        # Extra axonal water diffusivities 
        extra_diffusivities = np.random.uniform(low=self.config['synthetic_data_parameters']['extra_diff_lower'], high=self.config['synthetic_data_parameters']['extra_diff_upper'], size=2)

        prmts = {
            't2_intra': np.random.uniform(low=self.config['synthetic_data_parameters']['t2_intra_lower'], high=self.config['synthetic_data_parameters']['t2_intra_upper']),
            't2_extra': np.random.uniform(low=self.config['synthetic_data_parameters']['t2_extra_lower'], high=self.config['synthetic_data_parameters']['t2_extra_upper']),
            't2_dot': self.config['synthetic_data_parameters']['t2_dot'],
            't2_gm': np.random.uniform(low=self.config['synthetic_data_parameters']['t2_gm_lower'], high=self.config['synthetic_data_parameters']['t2_gm_upper']),
            't2_csf': self.config['synthetic_data_parameters']['t2_csf'],
            'intra_dpar': np.random.uniform(low=self.config['synthetic_data_parameters']['intra_dpar_lower'], high=self.config['synthetic_data_parameters']['intra_dpar_upper']),
            'intra_dperp': np.random.uniform(low=self.config['synthetic_data_parameters']['intra_dperp_lower'], high=self.config['synthetic_data_parameters']['intra_dperp_upper']),
            'extra_dpar': np.max(extra_diffusivities),
            'extra_dperp': np.min(extra_diffusivities),
            'csf_diso': np.random.uniform(low=self.config['synthetic_data_parameters']['csf_diso_lower'], high=self.config['synthetic_data_parameters']['csf_diso_upper']), # CHANGE TO CONFIG
            'gm_diso': np.random.uniform(low=self.config['synthetic_data_parameters']['gm_diso_lower'], high=self.config['synthetic_data_parameters']['gm_diso_upper']),
            'dot_diso': self.config['synthetic_data_parameters']['dot_diso'],
            'relative_vf_intra': relative_vf_intra,
            'relative_vf_extra': relative_vf_extra, 
            'relative_vf_dot': self.config['synthetic_data_parameters']['relative_vf_dot'],
            'partial_volume_0': partial_volume_0, # wm
            'partial_volume_1': vf_gm, # gm
            'partial_volume_2': vf_csf,  # csf
            'snr': np.random.uniform(low=low_snr, high=high_snr)
        }
        return prmts




    def single_synthetic_signal(self, scheme_highb, odf_sh_coeff, list_idxs_sh_coeffs, idxs_5000, idxs_10000, M_5000, M_10000, R):
        # get random biophysical parameters
        prmts = self.get_random_parameters()   
        # get random ODF coefficients                       
        theodfcoeffs = self.get_random_odf(odf_sh_coeff, list_idxs_sh_coeffs)
        
        '''DOT'''
        E_ball = np.exp(-scheme_highb.b.values*prmts['dot_diso'])

        '''GM'''
        E_ball_gm = np.exp(-scheme_highb.b.values*prmts['gm_diso'])

        '''CSF'''
        E_ball_csf = np.exp(-scheme_highb.b.values*prmts['csf_diso'])
        
        '''INTRA'''
        E_intra = zeppelin.zeppelin_signal(
            block=scheme_highb,
                odf=theodfcoeffs,
                lambda_parallel=prmts['intra_dpar'],
                lambda_perpendicular=prmts['intra_dperp'],
                matrix_5000=M_5000, 
                matrix_10000=M_10000 
            )

        '''EXTRA'''
        E_extra = zeppelin.zeppelin_signal(
                block=scheme_highb,
                odf=theodfcoeffs,
                lambda_parallel=prmts['extra_dpar'],
                lambda_perpendicular=prmts['extra_dperp'],
                matrix_5000=M_5000, 
                matrix_10000=M_10000 
            )
        
        # Calculate the ground truth signal high b-value signals without noise
        signal_gt =  prmts['partial_volume_0']*prmts['relative_vf_intra']*E_intra*np.exp(-self.te/prmts['t2_intra'])
        signal_gt += prmts['partial_volume_0']*prmts['relative_vf_extra']*E_extra*np.exp(-self.te/prmts['t2_extra'])
        signal_gt += prmts['partial_volume_0']*prmts['relative_vf_dot']*E_ball*np.exp(-self.te/prmts['t2_dot'])
        signal_gt += prmts['partial_volume_1']*E_ball_gm*np.exp(-self.te/prmts['t2_gm'])
        signal_gt += prmts['partial_volume_2']*E_ball_csf*np.exp(-self.te/prmts['t2_csf'])
        
        
        # Calculate the b=0 where all E is 1
        signal_0 = prmts['partial_volume_0']*prmts['relative_vf_intra']*np.exp(-self.te/prmts['t2_intra'])
        signal_0 += prmts['partial_volume_0']*prmts['relative_vf_extra']*np.exp(-self.te/prmts['t2_extra'])
        signal_0 += prmts['partial_volume_0']*prmts['relative_vf_dot']*np.exp(-self.te/prmts['t2_dot'])
        signal_0 += prmts['partial_volume_1']*np.exp(-self.te/prmts['t2_gm'])
        signal_0 += prmts['partial_volume_2']*np.exp(-self.te/prmts['t2_csf'])

        # Add noise to the signal according to the noise distribution specified in the config
        if self.config['real_data_preprocessing']['noise_distribution'] == 'rician':
            noisestd = signal_0/prmts['snr']
            noise_real_b0 = np.random.normal(0, noisestd, signal_0.shape)
            noise_imag_b0 = np.random.normal(0, noisestd, signal_0.shape)
            dwi_signal_0 = np.sqrt((signal_0 + noise_real_b0)**2 + noise_imag_b0**2)

            dwi_signal_0 = signal_0 + np.mean(noisestd * np.random.randn(9))
            noise_real = np.random.normal(0, noisestd, signal_gt.shape)
            noise_imag = np.random.normal(0, noisestd, signal_gt.shape)
            dwi_signal = np.sqrt((signal_gt + noise_real)**2 + noise_imag**2)
            
            dwi_signal = dwi_signal/dwi_signal_0

        elif self.config['real_data_preprocessing']['noise_distribution'] == 'gaussian':
            noisestd = signal_0/prmts['snr']
            dwi_signal_0 = signal_0 + np.mean(noisestd * np.random.randn(9))
            dwi_signal = signal_gt + noisestd * np.random.randn(len(signal_gt))
            
            dwi_signal = dwi_signal/dwi_signal_0
        elif self.config['real_data_preprocessing']['noise_distribution'] == 'none':
            dwi_signal = signal_gt/signal_0
        else:
            raise ValueError(f"Invalid noise distribution: {self.config['real_data_preprocessing']['noise_distribution']}")
        
        # Fit the spherical harmonics to the dwi signal at high b-values
        coeffs_5000 = shfit.fit_sh(dwi_signal[idxs_5000], M_5000, R,lambda_reg=0)
        coeffs_10000 = shfit.fit_sh(dwi_signal[idxs_10000], M_10000, R,lambda_reg=0)

        # Calculate the l-band power spectra for both b-values
        lbps_5000 = sh.compute_l_band_power_spectra(coeffs_5000,self.sh_order)
        lbps_10000 = sh.compute_l_band_power_spectra(coeffs_10000,self.sh_order)

        # root-square l-band power spectra ratios
        rslbpsr = (lbps_10000/lbps_5000)**0.5
        zonal_ratios = rslbpsr

        # Spherical variances 
        sphvar_5000 = sh.compute_sphvar(coeffs_5000)
        sphvar_10000 = sh.compute_sphvar(coeffs_10000)
        sphvar = np.array([sphvar_5000,sphvar_10000])

        # root-sqiare l-band spectra
        rslbps_5000 = lbps_5000**0.5
        rslbps_10000 = lbps_10000**0.5
            
        # Spherical means 
        sphm = np.array([sh.compute_sphmean(coeffs_5000),sh.compute_sphmean(coeffs_10000)])

        # Caluculate the Axonal signal fraction
        C = prmts['partial_volume_0']*prmts['relative_vf_intra']*np.exp(-self.te/prmts['t2_intra'])
        asf = (C/signal_0).clip(0,1.0) 
            
        # Save axonal and radial diffusivity 
        labels = np.array([prmts['intra_dpar'],prmts['intra_dperp']])
            
        return dwi_signal, zonal_ratios, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts


    def synthetic_simulation_parallel(self): 
        '''
        Runs the synthetic simulation in prallel. 
        Uses function single_synthetic_signal in parallel to generate full data set 
        '''
        # Get the acqusition scheme 
        scheme_df = pd.read_pickle(self.config['data_paths']['scheme'])
        scheme_df = scheme_df[(scheme_df.te == self.te)].copy()
        df_b_mask = (scheme_df.b==self.config['real_data_acquisition']['high_bmin']) | (scheme_df.b==self.config['real_data_acquisition']['high_bmax']) 
        scheme_highb = scheme_df[df_b_mask]
        scheme_highb = scheme_highb.reset_index(drop=True)
        
        odf_sh_coeff, list_idxs_sh_coeffs = self.get_odfs() # Obtain the ODFs
        
        # Find the indexs of the dwi-signal of the corresponding b_val 
        idxs_5000 = scheme_highb.b== self.config['real_data_acquisition']['high_bmin']
        idxs_10000 = scheme_highb.b== self.config['real_data_acquisition']['high_bmax']
        xyz_5000 = np.c_[scheme_highb.x.values[idxs_5000],scheme_highb.y.values[idxs_5000],scheme_highb.z.values[idxs_5000]]
        xyz_10000 = np.c_[scheme_highb.x.values[idxs_10000],scheme_highb.y.values[idxs_10000],scheme_highb.z.values[idxs_10000]]

        # Create spherical harmonics design matrices for HIGH b-value only
        M_5000 = sh.sh_matrix(self.sh_order, xyz_5000) # Use for matrix in zeppelin_signal function
        M_10000 = sh.sh_matrix(self.sh_order, xyz_10000) # Use for matrix in zeppelin_signal function
        R = sh.create_regularization_matrix(self.sh_order,type='lb')

        # Run the simulation of the signals in parallel
        list_synthetic_signals = Parallel(n_jobs=10, max_nbytes='10M')(delayed(self.single_synthetic_signal)(scheme_highb, odf_sh_coeff, list_idxs_sh_coeffs, idxs_5000, idxs_10000, M_5000, M_10000, R) for _ in range(int(self.config["synthetic_data_parameters"]["N_signals"])))
        
        # Transforms the parameters to numpy array 
        dwi_signal, alphas, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts = zip(*list_synthetic_signals)
        
        dwi_signal = np.array(dwi_signal)
        alphas = np.array(alphas)
        rslbps_5000 = np.array(rslbps_5000)
        rslbps_10000 = np.array(rslbps_10000)
        sphm = np.array(sphm)
        sphvar = np.array(sphvar)
        labels = np.array(labels)
        asf = np.array(asf)
     
        
        return dwi_signal, alphas, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts



    def save_simulation(self, dwi_signal, alphas, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts):
        # Save as npy file
        save_path = Path(self.config['mapping']['synthetic_data'])
        save_path.mkdir(parents=True, exist_ok=True)
        name_of_simulation = f'train_data_{self.config["synthetic_data_parameters"]["N_signals"]}_synthetic_signals_snr_{self.config["synthetic_data_parameters"]["snr"]}_dirichlet_{self.config["synthetic_data_parameters"]["alphas_dirichlet"]}'
        np.save(os.path.join(save_path, f'{name_of_simulation}_dwi_signal.npy'), dwi_signal)
        np.save(os.path.join(save_path, f'{name_of_simulation}_alphas.npy'), alphas)
        np.save(os.path.join(save_path, f'{name_of_simulation}_rslbps_5000.npy'), rslbps_5000)
        np.save(os.path.join(save_path, f'{name_of_simulation}_rslbps_10000.npy'), rslbps_10000)
        np.save(os.path.join(save_path, f'{name_of_simulation}_sphm.npy'), sphm)
        np.save(os.path.join(save_path, f'{name_of_simulation}_sphvar.npy'), sphvar)
        np.save(os.path.join(save_path, f'{name_of_simulation}_labels.npy'), labels)
        np.save(os.path.join(save_path, f'{name_of_simulation}_asf.npy'), asf)
        
        print(f'Saved {self.config["synthetic_data_parameters"]["N_signals"]} synthetic signals to {save_path}')

        filename = f'{name_of_simulation}_prmts.json'
        with open(os.path.join(save_path,filename), 'w') as f:
            json.dump(prmts, f)
        
        return None

    def generate(self): 
        dwi_signal, alphas, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts = self.synthetic_simulation_parallel()
        self.save_simulation(dwi_signal, alphas, rslbps_5000, rslbps_10000, sphm, sphvar, labels, asf, prmts)
        

        X_features= alphas
        if self.config["features_labels"]["sphm_high_b_lower"]:
            X_features = np.concatenate((X_features, sphm[:,0, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphm_high_b_upper"]:
            X_features = np.concatenate((X_features, sphm[:,1, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphvar_high_b_lower"]:
            X_features = np.concatenate((X_features, sphvar[:,0, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphvar_high_b_upper"]:
            X_features = np.concatenate((X_features, sphvar[:,1, np.newaxis]), axis=1)
        if self.config["features_labels"]["spectra_high_b_lower"]:
            X_features = np.concatenate((X_features, rslbps_5000), axis=1)
        if self.config["features_labels"]["spectra_high_b_upper"]:
            X_features = np.concatenate((X_features, rslbps_10000), axis=1)

        y_labels = labels.copy()
        y_labels[:,0] = y_labels[:,0]*1e10
        y_labels[:,1] = y_labels[:,1]*1e12
        if self.config["features_labels"]["predicted_asf"]: 
            y_labels = np.concatenate((y_labels,asf[:,np.newaxis]), axis=1)

        return X_features, y_labels