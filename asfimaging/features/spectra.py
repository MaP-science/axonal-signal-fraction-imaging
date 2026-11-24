import numpy as np
import pandas as pd
import nibabel as nib
from asfimaging.features.features import Features


class Spectra(Features):
    """Root-square l-band power spectra of the two high shells.

    b1 is the lower high shell (high_bmin) and b2 the upper one (high_bmax).
    """


    def calculate(self, data_RSLBPS):
        # Get only rslbps for high b-values
        scheme_df = self.scheme
        acquisition = self.config['real_data_acquisition']
        df_b_mask = (scheme_df.b==acquisition['high_bmin']) | (scheme_df.b==acquisition['high_bmax'])
        scheme_df = scheme_df[df_b_mask]
        data_highb_RSLBPS = data_RSLBPS[...,scheme_df.index]

        b1, b2 = np.sort(scheme_df.b.unique())
        zonal_orders = np.sort(scheme_df.zonal_order.unique())

        spectra = {}
        schemes = {}
        for name, bvalue in (('b1', b1), ('b2', b2)):
            columns = []
            rows = []
            for zo in zonal_orders:
                selected = ((scheme_df.zonal_order==zo) & (scheme_df.b==bvalue)).to_numpy()
                columns.append(np.abs(data_highb_RSLBPS[..., selected][..., 0]))
                row = scheme_df[selected].iloc[0]
                rows.append({'zonal_order': zo, 'te': row.te, 'dshell': row.dshell, 'b': bvalue})
            spectra[name] = np.nan_to_num(np.stack(columns, axis=-1))
            schemes[name] = pd.DataFrame(rows)

        return spectra['b1'], schemes['b1'], spectra['b2'], schemes['b2']

    def save(self, data_affine, spectra_b1, scheme_b1, spectra_b2, scheme_b2):
        directory = self.output_directory()
        nib.save(nib.Nifti1Image(spectra_b1, data_affine), directory / 'spectra_b1.nii.gz')
        scheme_b1.to_pickle(directory / 'spectra_b1_scheme.pkl')
        nib.save(nib.Nifti1Image(spectra_b2, data_affine), directory / 'spectra_b2.nii.gz')
        scheme_b2.to_pickle(directory / 'spectra_b2_scheme.pkl')
        return None
