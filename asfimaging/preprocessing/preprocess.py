from pathlib import Path
import numpy as np

from asfimaging.config import output_directory


class Preprocessor:

    def __init__(self, config):
        self.config = config
        self.slice_data_name = None
        self.slice_mask_name = None
      



    def extract_middle_slice(self, dmri, mask, plane="axial"):
        """
        Extract middle slice of a dMRI volume based on a mask.

        Parameters
        ----------
        plane : str
            'axial', 'coronal', or 'sagittal'

        Returns
        -------
        slice_dmri : np.ndarray
            Extracted 3D slice
        slice_mask : np.ndarray
            Corresponding 2D mask slice
        slice_index : int
            Index of the slice used
        """



        # --- Get mask coordinates ---
        coords = np.argwhere(mask > 0)

        if coords.size == 0:
            raise ValueError("Mask is empty")

        # --- Compute center of mass ---
        center = coords.mean(axis=0)

        # --- Select plane ---
        plane = plane.lower()

        if plane == "axial":
            idx = int(round(center[2]))
            slice_dmri = dmri[:, :, idx, np.newaxis,:]
            slice_mask = mask[:, :, idx, np.newaxis]

        elif plane == "coronal":
            idx = int(round(center[1]))
            slice_dmri = dmri[:, idx:idx + 1, :, :]
            slice_mask = mask[:, idx:idx + 1, :]

        elif plane == "sagittal":
            idx = int(round(center[0]))
            slice_dmri = dmri[idx:idx + 1, :, :, :]
            slice_mask = mask[idx:idx + 1, :, :]

        else:
            raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")

        return slice_dmri, slice_mask, idx

    def save(self,raw_data, mask, idx):
        """Save the slice under mapping.slices_dir (default <folder of data_paths.dwi>/results)."""
        def slice_path(source):
            path = Path(source)
            name = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
            return str(output_directory(self.config, 'slices_dir') / f'{name}_slice_{idx}.npy')

        self.slice_data_name = slice_path(self.config['data_paths']['dwi'])
        self.slice_mask_name = slice_path(self.config['data_paths']['mask'])
        np.save(self.slice_data_name, raw_data)
        np.save(self.slice_mask_name, mask)
