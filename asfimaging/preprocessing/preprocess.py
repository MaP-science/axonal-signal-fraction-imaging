import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


class preprocess: 

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
            slice_dmri = dmri[:, idx, :, np.newaxis,:]  
            slice_mask = mask[:, idx, :, np.newaxis]  

        elif plane == "sagittal":
            idx = int(round(center[0]))
            slice_dmri = dmri[idx, :, :, np.newaxis,:]
            slice_mask = mask[idx, :, :, np.newaxis]

        else:
            raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")

        return slice_dmri, slice_mask, idx

    def save(self,raw_data, mask, idx): 
        # save slices 
        base_data = self.config["data_paths"]["dwi"][:-7]  # remove ".nii.gz"
        new_name_data = f"{base_data}_slice_{idx}.npy"
        self.slice_data_name = new_name_data
        np.save(new_name_data, raw_data)
        base_mask = self.config["data_paths"]["mask"][:-7]  # remove ".nii.gz"
        new_name_mask = f"{base_mask}_slice_{idx}.npy"
        np.save(new_name_mask, mask)
        self.slice_mask_name = new_name_mask




    def load(self): 
        if self.slice_data_name is not None and self.slice_mask_name is not None: 
            raw_data = np.load(self.slice_data_name)
            mask = np.load(self.slice_mask_name)
            return raw_data, mask 
        else: 
            ValueError("Could not find a saved slice")

         
    def show_slice(self, slice_dmri, slice_mask, volume_index=0): 
        im = slice_dmri[...,volume_index] * slice_mask

        plt.imshow(im.T, cmap='gray', origin='lower')
        plt.title('Middle Slice of dMRI Volume with Mask Overlay')
        plt.axis('off')
        plt.show()
        return 