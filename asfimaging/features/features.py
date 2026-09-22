import json
import os
import numpy as np
import nibabel as nib
import pandas as pd
from abc import abstractmethod


class Features(object): 

    def __init__(self, config):
        self.config = config


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

    @abstractmethod
    def calculate(self, dwi_data, mask_data, scheme):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError("Subclasses must implement this method")
    
    def flatten_features(self, features, mask):
        idxs = mask>0
        print(idxs.shape)
        feature_mask = features[idxs]
        print(features.shape, feature_mask.shape)
        return feature_mask
    

    def unflatten_features(self, features_flattened, mask):
        idxsw = np.where(mask>0)
        features = np.zeros(mask.shape[:3])
        features[idxsw[0],idxsw[1],idxsw[2]] = features_flattened
        return features
