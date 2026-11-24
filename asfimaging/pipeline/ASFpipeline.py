import os
import json
import logging

from asfimaging.config import load_config, metrics_path, output_directory, require_files
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
from asfimaging.model.asf import ASFCalculator
from asfimaging.synthetic_data.generator import SyntheticSignalGenerator
from asfimaging.preprocessing.preprocess import Preprocessor

logger = logging.getLogger(__name__)


class ASFpipeline:

    def __init__(self,path_config):
        self.path_config = path_config
        self.config = self.get_config() 


    def get_config(self):
        return load_config(self.path_config)

    # -------------------------
    # Synthetic data
    # -------------------------

    def synthetic_train_data(self):
        # 1. Generate synthetic data (seed: synthetic_data_parameters.random_seed_train)
        generator = SyntheticSignalGenerator(self.config)
        X, y = generator.generate('train')
        return X, y

    def synthetic_test_data(self):
        # 1. Generate synthetic data (seed: synthetic_data_parameters.random_seed_test)
        generator = SyntheticSignalGenerator(self.config)
        X, y = generator.generate('test')
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
        # Generate the training split and train on it
        X, y = self.synthetic_train_data()
        return self.train(X, y)

    def evaluate_synthetic(self):
        """Evaluate the saved model on the synthetic test split (random_seed_test).

        Reports the forest targets and the calculated ASF (from predicted d_par,
        d_perp) against the simulated ground truth, and saves them as JSON at
        mapping.metrics_path (default: next to the model).
        """
        require_files(self.config, ('mapping', 'model_path'), ('mapping', 'scaler_path'))
        X, y = self.synthetic_test_data()
        inference = RFInference(self.config)
        metrics, predictions = inference.evaluate(X, y)

        # Calculated ASF from the predicted diffusivities (SI units) vs the true ASF
        generator = SyntheticSignalGenerator(self.config)
        predicted_si = predictions[:, :2] * [1e-10, 1e-12]
        asf_calculated = ASFCalculator(self.config).synthetic_calculate_asf(predicted_si, split='test')
        asf_true = np.load(generator.simulation_path('asf', 'test'))
        metrics['calculated_asf'] = inference.regression_metrics(asf_true, asf_calculated)

        result = {
            'experiment': self.config.get('experiment'),
            'model_path': self.config['mapping']['model_path'],
            'test_data': str(generator.simulation_path('dwi_signal', 'test')),
            'metrics': metrics,
        }
        path = metrics_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        return metrics
    
    # -------------------------
    # Features for real data
    # -------------------------

    def data_extractor(self, plane_slice = None): 
        require_files(self.config, ('data_paths', 'dwi'), ('data_paths', 'mask'))
        extract_data = Features(self.config)
        raw_data, raw_affine = extract_data.get_dwi()
        mask = extract_data.get_mask()
        if raw_data.ndim != 4 or mask.shape != raw_data.shape[:3]:
            raise ValueError("Expected 4D DWI and a mask matching its spatial shape")
        if not np.all(np.isfinite(mask)) or not np.any(mask > 0):
            raise ValueError("Mask must be finite and contain positive voxels")

        if plane_slice: 
            data_slice = Preprocessor(self.config)
            raw_data, mask, idx = data_slice.extract_middle_slice(raw_data, mask, plane_slice)
            axis = {'sagittal': 0, 'coronal': 1, 'axial': 2}[plane_slice.lower()]
            offset = np.zeros(3)
            offset[axis] = idx
            raw_affine = raw_affine.copy()
            raw_affine[:3, 3] += raw_affine[:3, :3] @ offset
            logger.info("Using the middle %s slice, index %d", plane_slice.lower(), idx)
            # save slices 
            data_slice.save(raw_data, mask, idx)
        

        return raw_data, raw_affine, mask

    def real_feature_extractor(self, raw_data, raw_affine, mask): 
        # Check file paths
        # Get raw data
        extract_data = Features(self.config)
        require_files(self.config, ('data_paths', 'scheme'))
        scheme = extract_data.get_scheme()
        if len(scheme) != raw_data.shape[-1]:
            raise ValueError("Scheme rows must match DWI volumes")

       
        # 1. Extract features and save them
        rslbps_extractor = RSLBPS(self.config, scheme)
        rslbps, rslbps_scheme = rslbps_extractor.calculate(raw_data, mask)
        rslbps_extractor.save(raw_affine, rslbps, rslbps_scheme)

        features = []

        if self.config["features_labels"]["zonal_ratios"]:
            zonal_ratios_extractor = ZonalRatios(self.config, rslbps_scheme)
            zonal_ratios, zonal_ratios_scheme = zonal_ratios_extractor.calculate(rslbps, mask)
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

        if not features:
            raise ValueError("Select at least one feature family in features_labels")
        x_ = np.concatenate(features, axis=1)
        return x_
    


    # -------------------------
    # Inference
    # -------------------------

    def predict_from_features(self, features):
        # 2. Predict labels
        require_files(self.config, ('mapping', 'model_path'), ('mapping', 'scaler_path'))
        rf_predictor = RFInference(self.config)
        labels = rf_predictor.inference(features)
        rf_predictor.save(labels)

        return labels  


    def predict_from_raw_data(self):
        raw_data, affine, mask = self.data_extractor()
        features = self.real_feature_extractor(raw_data, affine, mask)
        return self.predict_from_features(features)

    def get_premade_features(self):
        """Load voxel features in exactly the same order as fresh extraction."""
        require_files(self.config, ('data_paths', 'mask'))
        mask = Features(self.config).get_mask() > 0
        directory = output_directory(self.config, 'features_dir', create=False)
        flags = self.config['features_labels']
        features = []

        def append_image(filename, channel=None):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required cached feature file is missing: {path}")
            values = nib.load(path).get_fdata()
            if values.shape[:3] != mask.shape:
                raise ValueError(f"Cached feature shape does not match mask: {path}")
            if channel is not None:
                values = values[..., channel]
            values = values[mask]
            features.append(values.reshape(np.count_nonzero(mask), -1))

        if flags['zonal_ratios']:
            append_image('zonal_ratios.nii.gz')
        for prefix, filename in [('sphm', 'means.nii.gz'), ('sphvar', 'variances.nii.gz')]:
            for index, shell in enumerate(('lower', 'upper')):
                if flags[f'{prefix}_high_b_{shell}']:
                    append_image(filename, index)
        # Spectra.save names the lower high shell b1 and the upper one b2.
        if flags['spectra_high_b_lower']:
            append_image('spectra_b1.nii.gz')
        if flags['spectra_high_b_upper']:
            append_image('spectra_b2.nii.gz')
        if not features:
            raise ValueError("Select at least one feature family in features_labels")
        return np.concatenate(features, axis=1)

    # -------------------------
    # Save MAPS
    # -------------------------
    def save_maps(self, labels, *, raw_image=None, raw_affine=None, mask=None):
        # Get raw data
        if raw_image is None and raw_affine is None and mask is None:
            raw_image, raw_affine, mask = self.data_extractor()
        elif raw_image is None or raw_affine is None or mask is None:
            raise ValueError("Supply raw_image, raw_affine, and mask together")
        mask = np.asarray(mask)
        labels = np.asarray(labels)
        expected_columns = 3 if self.config['features_labels']['predicted_asf'] else 2
        if labels.shape != (np.count_nonzero(mask > 0), expected_columns):
            raise ValueError("Prediction shape does not match mask and enabled targets")
        require_files(self.config, ('data_paths', 'scheme'))

        # Save space for the labels maps
        dpar = np.zeros_like(mask, dtype=float)
        dperp = np.zeros_like(mask, dtype=float)
        asf_calculated = np.zeros_like(mask, dtype=float)
        asf_predicted = np.zeros_like(mask, dtype=float)

        idxs = np.where(mask > 0)
        

        dpar[idxs[0], idxs[1], idxs[2]] = labels[:,0]*1e-10
        dperp[idxs[0], idxs[1], idxs[2]] = labels[:,1]*1e-12


        # Calculate ASF 
        asf_calculated = ASFCalculator(self.config).real_calculate_asf(dpar, dperp, raw_image, mask)

        # Create nii maps 
        dpar_img = nib.Nifti1Image(dpar, raw_affine)
        dperp_img = nib.Nifti1Image(dperp, raw_affine)
        asf_cal_img = nib.Nifti1Image(asf_calculated, raw_affine)

        path_results = output_directory(self.config, 'maps_dir')

        nib.save(dpar_img, os.path.join(path_results, 'map_dpar.nii.gz'))
        nib.save(dperp_img, os.path.join(path_results, 'map_dperp.nii.gz'))
        nib.save(asf_cal_img, os.path.join(path_results, 'map_cal_asf.nii.gz'))


        if self.config["features_labels"]["predicted_asf"]:
            asf_predicted[idxs[0], idxs[1], idxs[2]] = labels[:,2]
            asf_pred_img = nib.Nifti1Image(asf_predicted, raw_affine)
            nib.save(asf_pred_img, os.path.join(path_results, 'map_pred_asf.nii.gz'))


        return dpar, dperp, asf_calculated, asf_predicted
