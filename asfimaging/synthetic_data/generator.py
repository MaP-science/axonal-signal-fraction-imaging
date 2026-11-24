import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from asfimaging.config import (SH_ORDER, check_sh_order, n_jobs, require_files, sh_fit_settings,
                               validate_config)

from asfimaging.shtools import sphericalharmonics as sh
from asfimaging.shtools import sphericalharmonicsfit as shfit
from asfimaging.sh_descoteaux import zeppelin_signal as zeppelin

logger = logging.getLogger(__name__)

SPLITS = ('train', 'test')


class SyntheticSignalGenerator:
    """Simulate signals on the two high shells b1 = high_bmin < b2 = high_bmax.

    The SH order is fixed at SH_ORDER (12): the ODF library, the Zeppelin
    kernel and the l-band features the forest is trained on all use it.
    """

    def __init__(self,config):
        self.config = validate_config(config)
        self.sh_order = check_sh_order(config)
        self.te = self.config['real_data_acquisition']['TE']
        self._seeds = {}
        snr =self.config['synthetic_data_parameters']['snr']
        if len(snr) == 6:
            warnings.warn(
                "synthetic_data_parameters.snr has 6 entries: using tissue-specific SNR ranges "
                "(WM-, GM- or CSF-dominated voxels use entries 0-1, 2-3 or 4-5; voxels with "
                "no dominant tissue use the WM range).",
                UserWarning,
            )

    def seed(self, split):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if split not in self._seeds:
            key = f'random_seed_{split}'
            seed = self.config['synthetic_data_parameters'].get(key)
            if seed is None:
                warnings.warn(f"synthetic_data_parameters.{key} is not set; the {split} data "
                              "will not be reproducible.", UserWarning)
                seed = np.random.SeedSequence().entropy % (2**32)
            self._seeds[split] = int(seed)
        return self._seeds[split]

    def n_signals(self, split):
        """N_signals for train; N_signals_test for test (falls back to N_signals)."""
        parameters = self.config['synthetic_data_parameters']
        if split == 'test' and parameters.get('N_signals_test') is not None:
            return int(parameters['N_signals_test'])
        return int(parameters['N_signals'])

    def get_odfs(self):
        """
        Sort the ODFs into bins based on their spherical variance.
        This is used to sample ODFs with different shapes for the synthetic signal generation.
        """
        # Select ODFs where there actually is a signal
        odf_sh_coeff = np.load(self.config['data_paths']['ODF'])
        odf_sh_coeff = odf_sh_coeff[odf_sh_coeff[...,0]>0]
        odf_sh_coeff = odf_sh_coeff[...,0:len(sh.compute_lm_ordervector(SH_ORDER))]
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

    def get_random_odf(self, rng, odf_sh_coeff, list_idxs_sh_coeffs):
        # Randomly select an ODF from the list of ODFs based on their spherical variance bins
        randomint = rng.integers(0, len(list_idxs_sh_coeffs))
        randomodf = rng.integers(0, len(list_idxs_sh_coeffs[randomint]))

        sh_coeffs_idx = list_idxs_sh_coeffs[randomint][randomodf]
        theodfcoeffs = odf_sh_coeff[sh_coeffs_idx,:]
        return theodfcoeffs


    def get_random_parameters(self, rng):
        parameters = self.config['synthetic_data_parameters']
        # Set the fractions of WM (intra, extra), GM and CSF
        vf_intra = rng.uniform(low=0.01, high=0.99)
        vf_extra, vf_gm, vf_csf  = rng.dirichlet(parameters['alphas_dirichlet'])
        not_intra = 1 - vf_intra

        # Set the volume fractions to amount not occupied by intra
        vf_extra = vf_extra*not_intra
        vf_gm = vf_gm*not_intra
        vf_csf = vf_csf*not_intra

        partial_volume_0 = vf_intra + vf_extra
        relative_vf_intra = vf_intra/partial_volume_0
        relative_vf_extra = vf_extra/partial_volume_0

        # Set the low and high snr uniformly
        snr = parameters['snr']
        low_snr, high_snr = snr[0], snr[1]

        if len(snr) == 6:  # Tissue-specific SNR ranges (see __init__ warning)
            if vf_gm > partial_volume_0 + vf_csf:
                low_snr, high_snr = snr[2], snr[3]
            elif vf_csf > partial_volume_0 + vf_gm:
                low_snr, high_snr = snr[4], snr[5]


        # Extra axonal water diffusivities
        extra_diffusivities = rng.uniform(low=parameters['extra_diff_lower'], high=parameters['extra_diff_upper'], size=2)

        prmts = {
            't2_intra': rng.uniform(low=parameters['t2_intra_lower'], high=parameters['t2_intra_upper']),
            't2_extra': rng.uniform(low=parameters['t2_extra_lower'], high=parameters['t2_extra_upper']),
            't2_dot': parameters['t2_dot'],
            't2_gm': rng.uniform(low=parameters['t2_gm_lower'], high=parameters['t2_gm_upper']),
            't2_csf': parameters['t2_csf'],
            'intra_dpar': rng.uniform(low=parameters['intra_dpar_lower'], high=parameters['intra_dpar_upper']),
            'intra_dperp': rng.uniform(low=parameters['intra_dperp_lower'], high=parameters['intra_dperp_upper']),
            'extra_dpar': float(np.max(extra_diffusivities)),
            'extra_dperp': float(np.min(extra_diffusivities)),
            'csf_diso': rng.uniform(low=parameters['csf_diso_lower'], high=parameters['csf_diso_upper']),
            'gm_diso': rng.uniform(low=parameters['gm_diso_lower'], high=parameters['gm_diso_upper']),
            'dot_diso': parameters['dot_diso'],
            'relative_vf_intra': float(relative_vf_intra),
            'relative_vf_extra': float(relative_vf_extra),
            'relative_vf_dot': parameters['relative_vf_dot'],
            'partial_volume_0': float(partial_volume_0), # wm
            'partial_volume_1': float(vf_gm), # gm
            'partial_volume_2': float(vf_csf),  # csf
            'snr': rng.uniform(low=low_snr, high=high_snr)
        }
        return prmts


    def single_synthetic_signal(self, rng, setup):
        scheme_highb = setup['scheme_highb']
        idxs_b1, idxs_b2 = setup['idxs_b1'], setup['idxs_b2']
        M_b1, M_b2 = setup['M_b1'], setup['M_b2']
        # get random biophysical parameters
        prmts = self.get_random_parameters(rng)
        # get random ODF coefficients
        theodfcoeffs = self.get_random_odf(rng, setup['odf_sh_coeff'], setup['list_idxs_sh_coeffs'])

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
                matrix_b1=M_b1,
                matrix_b2=M_b2
            )

        '''EXTRA'''
        E_extra = zeppelin.zeppelin_signal(
                block=scheme_highb,
                odf=theodfcoeffs,
                lambda_parallel=prmts['extra_dpar'],
                lambda_perpendicular=prmts['extra_dperp'],
                matrix_b1=M_b1,
                matrix_b2=M_b2
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

        # Add noise according to the configured distribution. The b0 used for
        # normalisation is the mean of n_b0 noisy b0 volumes, as in the scheme.
        n_b0 = setup['n_b0']
        noise_distribution = self.config['real_data_preprocessing']['noise_distribution']
        if noise_distribution == 'rician':
            noisestd = signal_0/prmts['snr']
            noise_real_b0 = rng.normal(0, noisestd, n_b0)
            noise_imag_b0 = rng.normal(0, noisestd, n_b0)
            dwi_signal_0 = np.mean(np.sqrt((signal_0 + noise_real_b0)**2 + noise_imag_b0**2))

            noise_real = rng.normal(0, noisestd, signal_gt.shape)
            noise_imag = rng.normal(0, noisestd, signal_gt.shape)
            dwi_signal = np.sqrt((signal_gt + noise_real)**2 + noise_imag**2)

            dwi_signal = dwi_signal/dwi_signal_0

        elif noise_distribution == 'gaussian':
            noisestd = signal_0/prmts['snr']
            dwi_signal_0 = signal_0 + np.mean(noisestd * rng.standard_normal(n_b0))
            dwi_signal = signal_gt + noisestd * rng.standard_normal(len(signal_gt))

            dwi_signal = dwi_signal/dwi_signal_0
        elif noise_distribution == 'none':
            dwi_signal = signal_gt/signal_0
        else:
            raise ValueError(f"Invalid noise distribution: {noise_distribution}")

        # Fit SH to the high-b signals exactly as for real data (same positivity and lambda)
        fit = setup['fit']
        coeffs_b1 = shfit.fit_sh_single(dwi_signal[idxs_b1], M_b1, setup['R'], fit['lambda_reg'],
                                        fit['positivity'], setup['M_positivity'], solver='SCS')
        coeffs_b2 = shfit.fit_sh_single(dwi_signal[idxs_b2], M_b2, setup['R'], fit['lambda_reg'],
                                        fit['positivity'], setup['M_positivity'], solver='SCS')

        # Calculate the l-band power spectra for both b-values
        lbps_b1 = sh.compute_l_band_power_spectra(coeffs_b1,self.sh_order)
        lbps_b2 = sh.compute_l_band_power_spectra(coeffs_b2,self.sh_order)

        # root-square l-band power spectra ratios
        # (0 where the b1 band power is 0, as for real data; see features.safe_divide)
        zonal_ratios = np.sqrt(np.divide(lbps_b2, lbps_b1, out=np.zeros_like(lbps_b2), where=lbps_b1 > 0))

        # Spherical variances
        sphvar = np.array([sh.compute_sphvar(coeffs_b1), sh.compute_sphvar(coeffs_b2)])

        # root-square l-band spectra
        rslbps_b1 = lbps_b1**0.5
        rslbps_b2 = lbps_b2**0.5

        # Spherical means
        sphm = np.array([sh.compute_sphmean(coeffs_b1),sh.compute_sphmean(coeffs_b2)])

        # Calculate the Axonal signal fraction
        C = prmts['partial_volume_0']*prmts['relative_vf_intra']*np.exp(-self.te/prmts['t2_intra'])
        asf = (C/signal_0).clip(0,1.0)

        # Save axonal and radial diffusivity
        labels = np.array([prmts['intra_dpar'],prmts['intra_dperp']])

        return dwi_signal, zonal_ratios, rslbps_b1, rslbps_b2, sphm, sphvar, labels, asf, prmts


    def simulate_batch(self, seed, indices, setup):
        """Simulate signals `indices`; signal i always uses the RNG seeded by [seed, i]."""
        return [self.single_synthetic_signal(np.random.default_rng([seed, int(i)]), setup)
                for i in indices]


    def synthetic_simulation_parallel(self, split='train'):
        '''
        Runs the synthetic simulation in parallel.
        Uses function single_synthetic_signal in parallel to generate full data set
        '''
        seed = self.seed(split)
        # Get the acqusition scheme
        scheme_df = pd.read_pickle(self.config['data_paths']['scheme'])
        scheme_df = scheme_df[(scheme_df.te == self.te)].copy()
        n_b0 = int(np.count_nonzero(scheme_df.b == 0))
        if n_b0 == 0:
            raise ValueError("The scheme has no b=0 volumes to normalise with")
        acquisition = self.config['real_data_acquisition']
        df_b_mask = (scheme_df.b==acquisition['high_bmin']) | (scheme_df.b==acquisition['high_bmax'])
        scheme_highb = scheme_df[df_b_mask]
        scheme_highb = scheme_highb.reset_index(drop=True)

        odf_sh_coeff, list_idxs_sh_coeffs = self.get_odfs() # Obtain the ODFs

        # Find the indexs of the dwi-signal of the corresponding b_val
        idxs_b1 = (scheme_highb.b == acquisition['high_bmin']).to_numpy()
        idxs_b2 = (scheme_highb.b == acquisition['high_bmax']).to_numpy()
        xyz_b1 = scheme_highb.loc[idxs_b1, ['x', 'y', 'z']].to_numpy()
        xyz_b2 = scheme_highb.loc[idxs_b2, ['x', 'y', 'z']].to_numpy()

        # SH design matrices for the high shells (used by zeppelin_signal and the fit)
        setup = {
            'scheme_highb': scheme_highb,
            'odf_sh_coeff': odf_sh_coeff,
            'list_idxs_sh_coeffs': list_idxs_sh_coeffs,
            'idxs_b1': idxs_b1,
            'idxs_b2': idxs_b2,
            'M_b1': sh.sh_matrix(self.sh_order, xyz_b1),
            'M_b2': sh.sh_matrix(self.sh_order, xyz_b2),
            'R': sh.create_regularization_matrix(self.sh_order,type='lb'),
            'M_positivity': sh.sh_matrix(self.sh_order, sh._positivity_directions),
            'n_b0': n_b0,
            'fit': sh_fit_settings(self.config),
        }

        # Run the simulation in batches so the shared setup is sent once per batch
        N = self.n_signals(split)
        workers = n_jobs(self.config, 'synthetic_data_parameters')
        n_batches = max(1, min(N, 8 * (os.cpu_count() if workers < 0 else workers)))
        batches = np.array_split(np.arange(N), n_batches)
        results = Parallel(n_jobs=workers, max_nbytes='10M')(
            delayed(self.simulate_batch)(seed, batch, setup) for batch in batches)
        list_synthetic_signals = [signal for batch in results for signal in batch]

        # Transforms the parameters to numpy array
        dwi_signal, zonal_ratios, rslbps_b1, rslbps_b2, sphm, sphvar, labels, asf, prmts = zip(*list_synthetic_signals)

        return (np.array(dwi_signal), np.array(zonal_ratios), np.array(rslbps_b1),
                np.array(rslbps_b2), np.array(sphm), np.array(sphvar), np.array(labels),
                np.array(asf), list(prmts))


    def simulation_name(self, split='train'):
        parameters = self.config["synthetic_data_parameters"]
        return (f'{split}_data_seed_{self.seed(split)}_{self.n_signals(split)}_synthetic_signals'
                f'_snr_{parameters["snr"]}_dirichlet_{parameters["alphas_dirichlet"]}')

    def simulation_path(self, suffix, split='train'):
        """Path of a saved simulation array, e.g. suffix='dwi_signal'."""
        return Path(self.config['mapping']['synthetic_data']) / f'{self.simulation_name(split)}_{suffix}.npy'


    def save_simulation(self, split, dwi_signal, zonal_ratios, rslbps_b1, rslbps_b2, sphm, sphvar, labels, asf, prmts):
        # Save as npy file
        save_path = Path(self.config['mapping']['synthetic_data'])
        save_path.mkdir(parents=True, exist_ok=True)
        arrays = {'dwi_signal': dwi_signal, 'zonal_ratios': zonal_ratios, 'rslbps_b1': rslbps_b1,
                  'rslbps_b2': rslbps_b2, 'sphm': sphm, 'sphvar': sphvar, 'labels': labels, 'asf': asf}
        for suffix, array in arrays.items():
            np.save(self.simulation_path(suffix, split), array)

        logger.info('Saved %d %s synthetic signals to %s', self.n_signals(split), split, save_path)

        with open(save_path / f'{self.simulation_name(split)}_prmts.json', 'w') as f:
            json.dump(prmts, f)

        return None

    def generate(self, split='train'):
        require_files(self.config, ('data_paths', 'scheme'), ('data_paths', 'ODF'))
        simulation = self.synthetic_simulation_parallel(split)
        self.save_simulation(split, *simulation)
        dwi_signal, zonal_ratios, rslbps_b1, rslbps_b2, sphm, sphvar, labels, asf, prmts = simulation


        X_features = zonal_ratios if self.config['features_labels']['zonal_ratios'] else np.empty((len(labels), 0))
        if self.config["features_labels"]["sphm_high_b_lower"]:
            X_features = np.concatenate((X_features, sphm[:,0, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphm_high_b_upper"]:
            X_features = np.concatenate((X_features, sphm[:,1, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphvar_high_b_lower"]:
            X_features = np.concatenate((X_features, sphvar[:,0, np.newaxis]), axis=1)
        if self.config["features_labels"]["sphvar_high_b_upper"]:
            X_features = np.concatenate((X_features, sphvar[:,1, np.newaxis]), axis=1)
        if self.config["features_labels"]["spectra_high_b_lower"]:
            X_features = np.concatenate((X_features, rslbps_b1), axis=1)
        if self.config["features_labels"]["spectra_high_b_upper"]:
            X_features = np.concatenate((X_features, rslbps_b2), axis=1)

        if X_features.shape[1] == 0:
            raise ValueError("Select at least one feature family in features_labels")
        y_labels = labels.copy()
        y_labels[:,0] = y_labels[:,0]*1e10
        y_labels[:,1] = y_labels[:,1]*1e12
        if self.config["features_labels"]["predicted_asf"]:
            y_labels = np.concatenate((y_labels,asf[:,np.newaxis]), axis=1)

        return X_features, y_labels
