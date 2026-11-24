"""Evaluate the trained model on a synthetic test set (random_seed_test)."""
import argparse
import logging

from asfimaging.config import metrics_path
from asfimaging.pipeline.ASFpipeline import ASFpipeline


def evaluate(config_name):
    pipeline = ASFpipeline(config_name)
    return pipeline, pipeline.evaluate_synthetic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a JSON configuration')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pipeline, metrics = evaluate(args.config)
    print(f"{'target':<16}{'MAE':>12}{'RMSE':>12}{'bias':>12}{'R^2':>9}  unit")
    for name, m in metrics.items():
        print(f"{name:<16}{m['mae']:>12.4g}{m['rmse']:>12.4g}{m['bias']:>12.4g}{m['r2']:>9.3f}  {m['unit']}")
    print(f'Saved metrics to {metrics_path(pipeline.config)}')


if __name__ == '__main__':
    main()
