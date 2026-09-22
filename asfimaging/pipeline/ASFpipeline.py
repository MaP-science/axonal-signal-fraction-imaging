import os
import json
from pyexpat import features
import numpy as np
import nibabel as nib

from asfimaging.features.features import Features
from asfimaging.features.rslbps import RSLBPS
from asfimaging.features.means import Means
from asfimaging.features.variances import Variances
from asfimaging.features.spectra import Spectra
from asfimaging.features.zonal_ratios import ZonalRatios

from asfimaging.model.RF_train import RFTrain
from asfimaging.model.RF_inference import RFInference
from asfimaging.model.asf import calculate_ASF
from asfimaging.synthetic_data.generator import SyntheticSignalGenerator
from asfimaging.preprocessing.preprocess import preprocess


class ASFpipeline:

    def __init__(self,path_config):
        self.path_config = path_config
        self.config = self.get_config() 


    def get_config(self):
        config = json.load(open(self.path_config, 'r'))
        return config

    # -------------------------
    # Synthetic data
    # -------------------------

    def synthetic_train_data(self): 
        # 1. Generate synthetic data
        generator = SyntheticSignalGenerator(self.config)
        X, y = generator.generate()
        return X, y
    
    def synthetic_test_data(self): 
        # 1. Generate synthetic data
        generator = SyntheticSignalGenerator(self.config)
        X, y = generator.generate()
        return X, y


    # -------------------------
    # Training
    # -------------------------

    def train(self, X, y):
        # 2. Build model
        rf_trainer = RFTrain(self.config)

        # 3. Train model
        trained_RF_model, fitted_scaler = rf_trainer.train(X, y)

        return trained_RF_model, fitted_scaler
    

    def train_synthetic(self):
        # 1. Generate synthetic data
        generator = SyntheticSignalGenerator(self.config)
        X, y = generator.generate()

        # 2. Build model
        rf_trainer = RFTrain(self.config)

        # 3. Train model
        trained_RF_model, fitted_scaler = rf_trainer.train(X, y)

        return trained_RF_model, fitted_scaler
    
    # -------------------------
    # Features for real data
    # -------------------------

    def data_extractor(self, plane_slice = None): 
        extract_data = Features(self.config)
        raw_data, raw_affine = extract_data.get_dwi()
        mask = extract_data.get_mask()

        if plane_slice: 
            data_slice = preprocess(self.config)
            raw_data, mask, idx = data_slice.extract_middle_slice(raw_data, mask, plane_slice)
            print(f"Chosing the middle slice index {idx}")
            print(raw_data.shape)
            print(mask.shape)
            # save slices 
            data_slice.save(raw_data, mask, idx)
        

        return raw_data, raw_affine, mask

    def real_feature_extractor(self, raw_data, raw_affine, mask): 
        # Check file paths
        # Get raw data
        extract_data = Features(self.config)
        scheme = extract_data.get_scheme()


       
        # 1. Extract features and save them
        rslbps_extractor = RSLBPS(self.config, scheme)
        rslbps, rslbps_scheme = rslbps_extractor.calculate(raw_data, mask)
        rslbps_extractor.save(raw_affine, rslbps, rslbps_scheme)

        features = []

        if self.config["features_labels"]["zonal_ratios"]:
            zonal_ratios_extractor = ZonalRatios(self.config, rslbps_scheme)
            zonal_ratios, zonal_ratios_scheme = zonal_ratios_extractor.calculate(rslbps)
            zonal_ratios_extractor.save(raw_affine, zonal_ratios, zonal_ratios_scheme)
            zonal_ratios = extract_data.flatten_features(zonal_ratios, mask)
            features.append(zonal_ratios)
        if self.config["features_labels"]["sphm_high_b_lower"] or self.config["features_labels"]["sphm_high_b_upper"]:
            sphm_extractor = Means(self.config, scheme)
            sphm, sphm_scheme = sphm_extractor.calculate(raw_data, mask)
            sphm_extractor.save(raw_affine, sphm, sphm_scheme)
            if self.config["features_labels"]["sphm_high_b_lower"]:
                sphm_lower = extract_data.flatten_features(sphm[...,0], mask)
                features.append(sphm_lower[:,np.newaxis])
            if self.config["features_labels"]["sphm_high_b_upper"]:
                sphm_upper = extract_data.flatten_features(sphm[...,1], mask)
                features.append(sphm_upper[:,np.newaxis])
        if self.config["features_labels"]["sphvar_high_b_lower"] or self.config["features_labels"]["sphvar_high_b_upper"]:
            sphvar_extractor = Variances(self.config, scheme)
            sphvar, sphvar_scheme = sphvar_extractor.calculate(raw_data, mask)
            sphvar_extractor.save(raw_affine, sphvar, sphvar_scheme)
            if self.config["features_labels"]["sphvar_high_b_lower"]:
                sphvar_lower = extract_data.flatten_features(sphvar[...,0], mask)
                features.append(sphvar_lower[:,np.newaxis])
            if self.config["features_labels"]["sphvar_high_b_upper"]:
                sphvar_upper = extract_data.flatten_features(sphvar[...,1], mask)
                features.append(sphvar_upper[:,np.newaxis])
        if self.config["features_labels"]["spectra_high_b_lower"] or self.config["features_labels"]["spectra_high_b_upper"]:
            spectra_extractor = Spectra(self.config, rslbps_scheme)
            spectra_lower, spectra_lower_scheme, spectra_upper, spectra_upper_scheme = spectra_extractor.calculate(rslbps)
            spectra_extractor.save(raw_affine, spectra_lower, spectra_lower_scheme, spectra_upper, spectra_upper_scheme)
            if self.config["features_labels"]["spectra_high_b_lower"]:
                spectra_lower = extract_data.flatten_features(spectra_lower, mask)
                features.append(spectra_lower)
            if self.config["features_labels"]["spectra_high_b_upper"]:
                spectra_upper = extract_data.flatten_features(spectra_upper, mask)
                features.append(spectra_upper)

        x_ = np.concatenate(features, axis=1)
        return x_
    


    # -------------------------
    # Inference
    # -------------------------

    def predict_from_features(self, features):
        # 2. Predict labels
        rf_predictor = RFInference(self.config)
        labels = rf_predictor.inference(features)

        return labels  


    def predict_from_raw_data(self): # NOT DONE 
        # Get raw data
        extract_data = Features(self.config)
        raw_data = extract_data.get_raw_data()
        mask = extract_data.get_mask()
        scheme = extract_data.get_scheme()
        
        # 1. Extract features and save them
        rslbps_extractor = RSLBPS(self.config, scheme)
        rslbps, rslbps_scheme = rslbps_extractor.calculate(raw_data, mask)
        rslbps_extractor.save(rslbps, rslbps_scheme)

        features = np.array([])

        if self.config["features_labels"]["zonal_ratios"]:
            zonal_ratios_extractor = ZonalRatios(self.config)
            zonal_ratios, zonal_ratios_scheme = zonal_ratios_extractor.calculate(rslbps_scheme, rslbps, mask)
            zonal_ratios_extractor.save(zonal_ratios, zonal_ratios_scheme)
            zonal_ratios = extract_data.flatten_features(zonal_ratios, mask)
            features = np.concatenate((features, zonal_ratios), axis=0)
        if self.config["features_labels"]["sphm_high_b_lower"] or self.config["features_labels"]["sphm_high_b_upper"]:
            sphm_extractor = Means(self.config)
            sphm, sphm_scheme = sphm_extractor.calculate(raw_data, mask, scheme)
            sphm_extractor.save(sphm, sphm_scheme)
            if self.config["features_labels"]["sphm_high_b_lower"]:
                sphm_lower = extract_data.flatten_features(sphm[...,0], mask)
                features = np.concatenate((features, sphm_lower), axis=0)
            if self.config["features_labels"]["sphm_high_b_upper"]:
                sphm_upper = extract_data.flatten_features(sphm[...,1], mask)
                features = np.concatenate((features, sphm_upper), axis=0)
        if self.config["features_labels"]["sphvar_high_b_lower"] or self.config["features_labels"]["sphvar_high_b_upper"]:
            sphvar_extractor = Variances(self.config)
            sphvar, sphvar_scheme = sphvar_extractor.calculate(raw_data, mask, scheme)
            sphvar_extractor.save(sphvar, sphvar_scheme)
            if self.config["features_labels"]["sphvar_high_b_lower"]:
                sphvar_lower = extract_data.flatten_features(sphvar[...,0], mask)
                features = np.concatenate((features, sphvar_lower), axis=0)
            if self.config["features_labels"]["sphvar_high_b_upper"]:
                sphvar_upper = extract_data.flatten_features(sphvar[...,1], mask)
                features = np.concatenate((features, sphvar_upper), axis=0)
        if self.config["features_labels"]["spectra_high_b_lower"] or self.config["features_labels"]["spectra_high_b_upper"]:
            spectra_extractor = Spectra(self.config)
            spectra_lower, spectra_lower_scheme, spectra_upper, spectra_upper_scheme = spectra_extractor.calculate(rslbps, mask, scheme)
            spectra_extractor.save(spectra_lower, spectra_lower_scheme, spectra_upper, spectra_upper_scheme)
            if self.config["features_labels"]["spectra_high_b_lower"]:
                spectra_lower = extract_data.flatten_features(spectra_lower, mask)
                features = np.concatenate((features, spectra_lower), axis=0)
            if self.config["features_labels"]["spectra_high_b_upper"]:
                spectra_upper = extract_data.flatten_features(spectra_upper, mask)
                features = np.concatenate((features, spectra_upper), axis=0)

        # 2. Predict ASF
        rf_predictor = RFInference(features, self.config)
        labels = rf_predictor.inference()

        return labels  


    def get_premade_features(self):
        # Search for premade features in the config file
        zonal_ratios_filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'zonal_ratios.nii.gz')
        means_filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'means.nii.gz')
        variances_filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'variances.nii.gz')
        spectra_b1_filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b1.nii.gz')
        spectra_b2_filename = os.path.join(self.config['data_paths']['real_data'], 'features', 'spectra_b2.nii.gz')

        features_ = []
        if os.path.exists(zonal_ratios_filename) and self.config["features_labels"]["zonal_ratios"]:
            zonal_ratios = nib.load(zonal_ratios_filename).get_fdata()
            features_.append(np.asarray(zonal_ratios))
        if os.path.exists(means_filename) and (self.config["features_labels"]["sphm_high_b_lower"] or self.config["features_labels"]["sphm_high_b_upper"]):
            means = nib.load(means_filename).get_fdata()
            if self.config["features_labels"]["sphm_high_b_lower"]:
                means = means[...,0]
            if self.config["features_labels"]["sphm_high_b_upper"]:
                means = means[...,1]
            features_.append(np.asarray(means))
        if os.path.exists(variances_filename) and (self.config["features_labels"]["sphvar_high_b_lower"] or self.config["features_labels"]["sphvar_high_b_upper"]):
            variances = nib.load(variances_filename).get_fdata()
            if self.config["features_labels"]["sphvar_high_b_lower"]:
                variances = variances[...,0]
            if self.config["features_labels"]["sphvar_high_b_upper"]:
                variances = variances[...,1]
            features_.append(np.asarray(variances))
        if os.path.exists(spectra_b1_filename) and self.config["features_labels"]["spectra_high_b_lower"]:
            spectra_b1 = nib.load(spectra_b1_filename).get_fdata()
            features_.append(np.asarray(spectra_b1))
        if os.path.exists(spectra_b2_filename) and self.config["features_labels"]["spectra_high_b_upper"]:
            spectra_b2 = nib.load(spectra_b2_filename).get_fdata()
            features_.append(np.asarray(spectra_b2))

        x_features = np.c_[features_]
        return x_features



    # -------------------------
    # Save MAPS
    # -------------------------
    def save_maps(self, labels): 
        # Get raw data
        extract_data = Features(self.config)
        raw_image, raw_affine = extract_data.get_dwi()
        mask = extract_data.get_mask()

        # Save space for the labels maps
        dpar = np.zeros_like(mask)
        dperp = np.zeros_like(mask)
        asf_calculated = np.zeros_like(mask)
        asf_predicted = np.zeros_like(mask)
        print(mask.shape)

        idxs = np.where(mask == 1)
        

        dpar[idxs[0], idxs[1], idxs[2]] = labels[:,0]*1e-10
        dperp[idxs[0], idxs[1], idxs[2]] = labels[:,1]*1e-12


        # Calculate ASF 
        asf_calculated = calculate_ASF(self.config).real_calculate_asf(dpar, dperp, raw_image)

        # Create nii maps 
        dpar_img = nib.Nifti1Image(dpar, raw_affine)
        dperp_img = nib.Nifti1Image(dperp, raw_affine)
        asf_cal_img = nib.Nifti1Image(asf_calculated, raw_affine)

        path_results = self.config["mapping"]["path_maps"]

        nib.save(dpar_img, os.path.join(path_results, 'map_dpar.nii.gz'))
        nib.save(dperp_img, os.path.join(path_results, 'map_dperp.nii.gz'))
        nib.save(asf_cal_img, os.path.join(path_results, 'map_cal_asf.nii.gz'))


        if self.config["features_labels"]["predicted_asf"]:
            asf_predicted[idxs[0], idxs[1], idxs[2]] = labels[:,2]
            asf_pred_img = nib.Nifti1Image(asf_predicted, raw_affine)
            nib.save(asf_pred_img, os.path.join(path_results, 'map_pred_asf.nii.gz'))


        return dpar, dperp, asf_calculated, asf_predicted





    # -------------------------
    # Visualize MAPS
    # -------------------------

    def visualize(dpar, dperp, asf_calculated, asf_predicted): 

        return None

