import numpy as np
import pandas as pd
import nibabel as nib
from asfimaging.features.features import Features, safe_divide



class ZonalRatios(Features):
    """Per zonal order, RSLBPS(b2) / RSLBPS(b1), with b1 < b2 the two high shells."""


    def calculate(self, data_RSLBPS, mask=None):
        # Get only rslbps for high b-values
        scheme_df = self.scheme
        acquisition = self.config['real_data_acquisition']
        df_b_mask = (scheme_df.b==acquisition['high_bmin']) | (scheme_df.b==acquisition['high_bmax'])
        scheme_df = scheme_df[df_b_mask]
        data_highb_RSLBPS = data_RSLBPS[...,scheme_df.index]

        b1, b2 = np.sort(scheme_df.b.unique())
        zonal_orders = np.sort(scheme_df.zonal_order.unique())

        columns = []
        rows = []
        for zo in zonal_orders:
            selected_b1 = ((scheme_df.zonal_order==zo) & (scheme_df.b==b1)).to_numpy()
            selected_b2 = ((scheme_df.zonal_order==zo) & (scheme_df.b==b2)).to_numpy()
            signal_b1 = np.abs(data_highb_RSLBPS[..., selected_b1][..., 0])
            signal_b2 = np.abs(data_highb_RSLBPS[..., selected_b2][..., 0])
            columns.append(safe_divide(signal_b2, signal_b1, mask,
                                       label=f'Zonal ratio (zonal order {zo})'))
            row = scheme_df[selected_b2].iloc[0]
            rows.append({'zonal_order': zo, 'te': row.te, 'b_numerator': b2, 'b_denominator': b1})

        ratios = np.stack(columns, axis=-1)
        scheme_ratios = pd.DataFrame(rows)

        return ratios, scheme_ratios


    def save(self, data_affine, zonal_ratios, scheme):
        directory = self.output_directory()
        nib.save(nib.Nifti1Image(zonal_ratios, data_affine), directory / 'zonal_ratios.nii.gz')
        scheme.to_pickle(directory / 'zonal_ratios_scheme.pkl')
        return None
