"""Generate and save a diffusion acquisition scheme."""
import argparse
import logging
from pathlib import Path

from asfimaging.config import load_config
from asfimaging.preprocessing.generate_scheme import SchemeGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to a JSON configuration')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate an existing scheme')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    config = load_config(args.config)
    path = Path(config['data_paths']['scheme'])
    if path.is_file() and not args.overwrite:
        print(f'Scheme already exists at {path}; use --overwrite to regenerate.')
        return
    generator = SchemeGenerator(config)
    scheme = generator.generate_scheme()
    print(f'Scheme saved to {generator.save(scheme)}')


if __name__ == '__main__':
    main()
