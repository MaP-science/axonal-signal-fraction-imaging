"""Predict parameter and ASF maps from diffusion MRI."""
import argparse
import logging

from asfimaging.config import require_files
from asfimaging.pipeline.ASFpipeline import ASFpipeline


def predict(config_name, use_features=False):
    pipeline = ASFpipeline(config_name)
    require_files(pipeline.config, ('mapping', 'model_path'), ('mapping', 'scaler_path'),
                  ('data_paths', 'dwi'), ('data_paths', 'mask'), ('data_paths', 'scheme'))
    raw_data, affine, mask = pipeline.data_extractor()
    if use_features:
        features = pipeline.get_premade_features()
    else:
        features = pipeline.real_feature_extractor(raw_data, affine, mask)
    labels = pipeline.predict_from_features(features)
    return pipeline.save_maps(labels, raw_image=raw_data, raw_affine=affine, mask=mask)


def parse_bool(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError("Expected 'true' or 'false'")
    return value.lower() == 'true'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a JSON configuration')
    parser.add_argument('cached', nargs='?', type=parse_bool, default=False,
                        help='Legacy positional cached-feature switch: true or false')
    parser.add_argument('--use-features', action='store_true', help='Read previously saved features')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    predict(args.config, args.cached or args.use_features)
    print('Saved parameter and ASF maps at the configured path.')


if __name__ == '__main__':
    main()
