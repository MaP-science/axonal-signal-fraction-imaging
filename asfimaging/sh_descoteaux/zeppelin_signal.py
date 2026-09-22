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


ODF_ORDER = 12

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


def zeppelin_signal(block, odf, lambda_parallel, lambda_perpendicular, matrix_5000, matrix_10000):
    """
    Compute the signal for the Zeppelin diffusion model.
    input:
        block: a data block containing b-values and gradient directions
        odf: spherical harmonic coefficients of the orientation distribution function (ODF)
        lambda_parallel: parallel diffusivity
        lambda_perpendicular: perpendicular diffusivity
        matrix_5000: precomputed matrix for b=5000 s/mm^2
        matrix_10000: precomputed matrix for b=10000 s/mm^2
    output:
        signal: computed signal for the given b-values and ODF
    """
    bvalues = np.asarray(block.b, dtype=float)
    bunique = np.unique(bvalues)
    orders = coefficient_orders(ODF_ORDER)[:, 0]
    signal = np.empty(len(bvalues), dtype=float)

    # Compute the signal for b=5000 s/mm^2
    bvalue = bunique[0]
    x = bvalue * (lambda_parallel - lambda_perpendicular)
    eigenvalues = (
        np.exp(-bvalue * lambda_perpendicular)
        * 2.0
        * np.pi
        * psi_coefficients(x)
    )
    signal_5000 = matrix_5000 @ (odf * eigenvalues[orders // 2])

    # Compute the signal for b=10000 s/mm^2
    bvalue = bunique[1]
    x = bvalue * (lambda_parallel - lambda_perpendicular)
    eigenvalues = (
        np.exp(-bvalue * lambda_perpendicular)
        * 2.0
        * np.pi
        * psi_coefficients(x)
    )
    signal_10000 = matrix_10000 @ (odf * eigenvalues[orders // 2])

    # Assign the computed signals to the appropriate indices
    signal[np.isclose(bvalues, bunique[0])] = signal_5000
    signal[np.isclose(bvalues, bunique[1])] = signal_10000

    return signal
