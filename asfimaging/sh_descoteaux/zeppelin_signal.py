"""Zeppelin Signal Model Utilities.

Author: Marco Pizzolato
Contact: mapiz@dtu.dk
Affiliation: Department of Applied Mathematics and Computer Science, Technical University of Denmark

This module provides helper routines for spherical-harmonic coefficients and
signal synthesis for the Zeppelin diffusion model.
"""

from __future__ import annotations

import numpy as np
from scipy import special


from asfimaging.config import SH_ORDER

ODF_ORDER = SH_ORDER  # fixed: the bundled ODFs and trained features use order 12

nodes, weights = np.polynomial.legendre.leggauss(60)



def validate_sh_order(order: int) -> None:
    if order < 0 or order % 2:
        raise ValueError("Spherical-harmonic order must be a non-negative even integer.")


def coefficient_orders(order: int) -> np.ndarray:
    validate_sh_order(order)
    return np.asarray(
        [(degree, m) for degree in range(0, order + 1, 2) for m in range(-degree, degree + 1)],
        dtype=int,
    )



def psi_coefficients(x: float) -> np.ndarray:    
    attenuation = np.exp(-x * nodes**2)
    return np.asarray([
        np.sum(weights * special.eval_legendre(order, nodes) * attenuation)
        for order in range(0, ODF_ORDER + 1, 2)
    ])


def _shell_signal(bvalue, odf, lambda_parallel, lambda_perpendicular, matrix, orders):
    x = bvalue * (lambda_parallel - lambda_perpendicular)
    eigenvalues = (
        np.exp(-bvalue * lambda_perpendicular)
        * 2.0
        * np.pi
        * psi_coefficients(x)
    )
    return matrix @ (odf * eigenvalues[orders // 2])


def zeppelin_signal(block, odf, lambda_parallel, lambda_perpendicular, matrix_b1, matrix_b2):
    """
    Compute the signal for the Zeppelin diffusion model on the two high shells.
    input:
        block: a data block containing b-values (s/m^2) and gradient directions
        odf: spherical harmonic coefficients of the orientation distribution function (ODF)
        lambda_parallel: parallel diffusivity (m^2/s)
        lambda_perpendicular: perpendicular diffusivity (m^2/s)
        matrix_b1: precomputed SH matrix for the lower high shell b1 (high_bmin)
        matrix_b2: precomputed SH matrix for the upper high shell b2 (high_bmax)
    output:
        signal: computed signal for the given b-values and ODF
    """
    bvalues = np.asarray(block.b, dtype=float)
    bunique = np.unique(bvalues)  # ascending: [b1, b2]
    if len(bunique) != 2:
        raise ValueError(f"Expected exactly two high b-value shells, found {len(bunique)}")
    orders = coefficient_orders(ODF_ORDER)[:, 0]
    signal = np.empty(len(bvalues), dtype=float)

    signal_b1 = _shell_signal(bunique[0], odf, lambda_parallel, lambda_perpendicular, matrix_b1, orders)
    signal_b2 = _shell_signal(bunique[1], odf, lambda_parallel, lambda_perpendicular, matrix_b2, orders)

    # Assign the computed signals to the appropriate indices
    signal[np.isclose(bvalues, bunique[0])] = signal_b1
    signal[np.isclose(bvalues, bunique[1])] = signal_b2

    return signal
