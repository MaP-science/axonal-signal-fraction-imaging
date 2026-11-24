import warnings

import numpy as np
import scipy.special as ssp
import pandas as pd

from asfimaging.features.features import safe_divide


class ASFCalculator:
    """ASF = spherical mean at b1 (high_bmin) / intra-axonal Zeppelin spherical mean.

    Assumes the extra-axonal, GM and CSF signal is negligible at b1.
    """

    # Below this b*(d_par - d_perp) the closed form is replaced by its Taylor limit.
    SMALL_X = 1e-8

    def __init__(self, config):
        self.config = config

    def load_scheme(self):
        scheme = pd.read_pickle(self.config["data_paths"]["scheme"])
        return scheme

    def signal_spherical_mean(self, b, d_par, d_perp):
        """Spherical mean of an axially symmetric tensor (d_par >= d_perp assumed).

        S = exp(-b d_perp) * sqrt(pi)/2 * erf(sqrt(x))/sqrt(x), x = b (d_par - d_perp).
        As x -> 0 the factor sqrt(pi)/2 * erf(sqrt(x))/sqrt(x) -> 1 - x/3 -> 1, so the
        isotropic limit is exp(-b d_perp). Pairs with d_par < d_perp are clipped to
        that limit with a warning.
        """
        d_par = np.asarray(d_par, dtype=float)
        d_perp = np.asarray(d_perp, dtype=float)
        x = b * (d_par - d_perp)
        negative = x < 0
        if np.any(negative):
            warnings.warn(
                f"{np.count_nonzero(negative)} value(s) have d_par < d_perp; "
                "using the isotropic limit d_par = d_perp for them.",
                RuntimeWarning,
            )
            x = np.where(negative, 0.0, x)
        small = x < self.SMALL_X
        x_safe = np.where(small, 1.0, x)
        anisotropy = np.where(
            small,
            1.0 - x / 3.0,
            np.sqrt(np.pi) / 2 * ssp.erf(np.sqrt(x_safe)) / np.sqrt(x_safe),
        )
        return np.exp(-b * d_perp) * anisotropy

    def high_b_scheme(self):
        """Columns of the saved synthetic dwi_signal: high shells at the configured TE."""
        scheme = self.load_scheme()
        acquisition = self.config["real_data_acquisition"]
        scheme = scheme[scheme.te == acquisition["TE"]]
        scheme = scheme[(scheme.b == acquisition["high_bmin"]) | (scheme.b == acquisition["high_bmax"])]
        return scheme.reset_index(drop=True)

    def load_synthetic_data(self, split="train"):
        """Mean b1 signal of the saved synthetic set in mapping.synthetic_data.

        The saved signals are already normalised by their noisy b0.
        """
        from asfimaging.synthetic_data.generator import SyntheticSignalGenerator

        path = SyntheticSignalGenerator(self.config).simulation_path("dwi_signal", split)
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic signals not found: {path}")
        data = np.load(path)
        idxs_b1 = (self.high_b_scheme().b == self.config["real_data_acquisition"]["high_bmin"]).to_numpy()
        return np.mean(data[..., idxs_b1], axis=-1)


    def load_real_data(self, data, mask=None):
        scheme = self.load_scheme()
        idxs_b0 = (scheme.b == 0).to_numpy()
        data_b0_mean = np.mean(data[...,idxs_b0],axis=-1)

        idxs_b1 = (scheme.b == self.config["real_data_acquisition"]["high_bmin"]).to_numpy()
        data_b1_mean = np.mean(data[...,idxs_b1],axis=-1)

        return safe_divide(data_b1_mean, data_b0_mean, mask, label='ASF b0 normalisation')

    def synthetic_calculate_asf(self, labels, split="train"):
        """labels: (N, 2) intra-axonal [d_par, d_perp] in m^2/s."""
        d_par = labels[:,0]
        d_perp = labels[:,1]

        sphmean = self.load_synthetic_data(split)
        C = self.signal_spherical_mean(self.config["real_data_acquisition"]["high_bmin"],d_par,d_perp)
        asf = np.nan_to_num(sphmean/C)
        asf = np.clip(asf, 0, 1)
        return asf

    def real_calculate_asf(self, dpar, dperp, data, mask=None):
        """dpar, dperp in m^2/s. Voxels outside `mask` (if given) are set to 0."""
        inside = np.ones(np.shape(dpar), dtype=bool) if mask is None else np.asarray(mask) > 0
        sphmean = self.load_real_data(data, mask)
        C = np.ones(np.shape(dpar))
        C[inside] = self.signal_spherical_mean(
            self.config["real_data_acquisition"]["high_bmin"], dpar[inside], dperp[inside])
        asf = np.zeros(np.shape(dpar))
        asf[inside] = np.nan_to_num(sphmean[inside] / C[inside])
        asf = np.clip(asf, 0, 1)
        return asf
