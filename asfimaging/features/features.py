import numpy as np
import nibabel as nib
import pandas as pd
import warnings

from asfimaging.config import output_directory


def safe_divide(numerator, denominator, mask=None, label='division'):
    """numerator / denominator, with 0 where the denominator is <= 0 or not finite
    (or the result is not finite) instead of NaN/inf (nan_to_num would give ~1.8e308).

    Warns with the number of affected voxels inside `mask` (a 3D spatial mask; all
    voxels if None). Trailing non-spatial axes are reduced with "any".
    """
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float),
                                                 np.asarray(denominator, dtype=float))
    valid = (denominator > 0) & np.isfinite(denominator) & np.isfinite(numerator)
    result = np.zeros(numerator.shape)
    np.divide(numerator, denominator, out=result, where=valid)
    valid &= np.isfinite(result)
    result[~valid] = 0.0
    invalid = ~valid
    if mask is not None:
        mask = np.asarray(mask) > 0
        invalid = invalid.reshape(mask.shape + (-1,)).any(axis=-1) & mask
    count = int(np.count_nonzero(invalid))
    if count:
        where = 'inside the mask' if mask is not None else ''
        warnings.warn(f"{label}: {count} value(s) {where} had a zero, negative or non-finite "
                      "denominator and were set to 0.", RuntimeWarning)
    return result


class Features:
    """Base class for the feature extractors and shared data loading.

    Subclasses (RSLBPS, Means, Variances, ZonalRatios, Spectra) receive the scheme
    describing their input and implement ``calculate`` and ``save``. The base class
    is also instantiated directly, without a scheme, to load the DWI, mask and
    scheme (``get_dwi``, ``get_mask``, ``get_scheme``) and to flatten features.
    """

    def __init__(self, config, scheme=None):
        self.config = config
        self.scheme = scheme


    def output_directory(self):
        """mapping.features_dir, default <folder of data_paths.dwi>/results/features."""
        return output_directory(self.config, 'features_dir')

    def get_dwi(self):
        dwi_path = self.config['data_paths']['dwi']
        dwi_data = nib.load(dwi_path)
        dwi_affine = dwi_data.affine
        dwi_data = dwi_data.get_fdata()
        return dwi_data, dwi_affine

    def get_mask(self):
        mask_path = self.config['data_paths']['mask']
        mask_data = nib.load(mask_path).get_fdata()
        return mask_data

    def get_scheme(self):
        scheme_path = self.config['data_paths']['scheme']
        scheme = pd.read_pickle(scheme_path)
        return scheme

    def calculate(self, *args, **kwargs):
        """Compute the feature maps; implemented by each subclass.

        Inputs depend on the feature: the 4D DWI and mask (RSLBPS, Means,
        Variances, with the acquisition scheme) or the RSLBPS maps (ZonalRatios,
        Spectra, with the RSLBPS scheme). Returns the maps and their scheme table(s).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement calculate")

    def save(self, *args, **kwargs):
        """Save the maps (NIfTI, with the DWI affine) and scheme table(s) to
        output_directory(); implemented by each subclass."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save")
    
    def flatten_features(self, features, mask):
        """Voxels inside the mask, in r, c, s order: (n_voxels, ...)."""
        return features[mask > 0]
