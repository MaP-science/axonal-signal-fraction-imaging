"""Generate an acquisition table while preserving input volume order."""

from pathlib import Path

import numpy as np
import pandas as pd

from asfimaging.config import require_files, validate_config
from asfimaging.shtools import sphericalharmonics as sh


class SchemeGenerator:
    def __init__(self, config):
        self.config = validate_config(config)

    def load_bvals_bvecs(self, bvals_path, bvecs_path):
        bvals = np.atleast_1d(np.loadtxt(bvals_path)).reshape(-1)
        bvecs = np.atleast_2d(np.loadtxt(bvecs_path))
        if bvecs.shape == (len(bvals), 3) and bvecs.shape != (3, len(bvals)):
            bvecs = bvecs.T
        if bvecs.shape != (3, len(bvals)):
            raise ValueError("bvecs must have shape (3, N) or (N, 3), matching bvals")
        if not np.all(np.isfinite(bvals)) or not np.all(np.isfinite(bvecs)):
            raise ValueError("bvals and bvecs must be finite")
        if np.any(bvals < 0):
            raise ValueError("bvals cannot be negative")
        # Conventional bval files use s/mm^2; the internal scheme uses s/m^2.
        if np.max(bvals) < 1e6:
            bvals = bvals * 1e6
        return bvals, bvecs

    def snap_bvalues(self, bvals):
        """Snap b-values within tolerance to the two configured shells and to b0."""
        values = np.asarray(bvals, dtype=float).copy()
        acquisition = self.config['real_data_acquisition']
        for shell in ('high_bmin', 'high_bmax'):
            selected = np.abs(values - acquisition[shell]) <= acquisition['tolerance_bval']
            values[selected] = acquisition[shell]
        values[np.abs(values) <= acquisition['tolerance_b0']] = 0
        return values

    def check_data_requirements(self, bvals, bvecs):
        acquisition = self.config['real_data_acquisition']
        order = acquisition['sh_order']
        if not isinstance(order, int) or order < 0 or order % 2:
            raise ValueError("sh_order must be a non-negative even integer")
        if bvecs.shape != (3, len(bvals)):
            raise ValueError("Number of bvals does not match bvecs")
        if not np.any(bvals == 0):
            raise ValueError("At least one b0 volume is required for normalization")
        if not 0 < acquisition['high_bmin'] < acquisition['high_bmax']:
            raise ValueError("Expected 0 < high_bmin < high_bmax")
        required = len(sh.compute_lm_ordervector(order))
        for key in ('high_bmin', 'high_bmax'):
            selected = bvals == acquisition[key]
            count = np.count_nonzero(selected)
            if count < required:
                raise ValueError(f"{key} requires at least {required} directions; found {count}")
            if np.any(np.linalg.norm(bvecs[:, selected], axis=0) == 0):
                raise ValueError(f"{key} contains a zero direction vector")
        return True

    def create_scheme(self, bvecs, bvals):
        acquisition = self.config['real_data_acquisition']
        _, shells = np.unique(bvals, return_inverse=True)
        direction_indices = np.zeros(len(bvals), dtype=int)
        for shell in np.unique(shells):
            selected = shells == shell
            _, direction_indices[selected] = np.unique(
                bvecs[:, selected].T, axis=0, return_inverse=True
            )
        return pd.DataFrame({
            'x': bvecs[0], 'y': bvecs[1], 'z': bvecs[2],
            'te': acquisition['TE'], 'd': acquisition['delta'],
            'D': acquisition['Delta'], 'b': bvals, 'dshell': shells,
            'dir': direction_indices, 'type': 'pgse',
        })

    def check_scheme_validity(self, scheme):
        return self.check_data_requirements(
            scheme.b.to_numpy(), scheme[['x', 'y', 'z']].to_numpy().T
        )

    def generate_scheme(self):
        require_files(self.config, ('data_paths', 'bvals'), ('data_paths', 'bvecs'))
        bvals, bvecs = self.load_bvals_bvecs(
            self.config['data_paths']['bvals'], self.config['data_paths']['bvecs']
        )
        bvals = self.snap_bvalues(bvals)
        self.check_data_requirements(bvals, bvecs)
        scheme = self.create_scheme(bvecs, bvals)
        self.check_scheme_validity(scheme)
        return scheme

    def save(self, scheme):
        path = Path(self.config['data_paths']['scheme'])
        path.parent.mkdir(parents=True, exist_ok=True)
        scheme.to_pickle(path)
        return path
