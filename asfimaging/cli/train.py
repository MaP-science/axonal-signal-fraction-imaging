"""Train a random forest and save its model and feature scaler."""
import argparse
import logging

from asfimaging.pipeline.ASFpipeline import ASFpipeline


def train_RF_regressor(config_name):
    return ASFpipeline(config_name).train_synthetic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a JSON configuration')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    train_RF_regressor(args.config)
    print('Saved the trained model and scaler at the configured paths.')


if __name__ == '__main__':
    main()
