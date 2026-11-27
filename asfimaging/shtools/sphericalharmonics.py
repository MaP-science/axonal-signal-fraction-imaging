# Project-original code: AGPL-3.0-only; see LICENSE at the repository root.
# The four functions marked below are adapted from DIPY (BSD-3-Clause).
# Copyright (c) 2008-2025, dipy developers. All rights reserved.
# Retain LICENSES/DIPY-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md with copies.
# The original DIPY source version is unknown; the notice is preserved from
# the DIPY 1.11.0 reference used to verify provenance.

import numpy as np
from importlib.resources import files
import scipy.special as sps

# Directions at which the relaxed SH positivity constraint is enforced
# (325 near-uniform unit vectors; project-original, see DATA_AND_MODEL_PROVENANCE.md).
with files('asfimaging').joinpath('shtools/data/directions_325_l_24.npy').open('rb') as _stream:
    _positivity_directions = np.load(_stream, allow_pickle=False)

# Adapted from dipy.core.geometry.cart2sphere; BSD-3-Clause.
def cart2sphere(x, y, z):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    Parameters
    ------------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    ---------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi



# Adapted from dipy.reconst.shm.real_sph_harm; BSD-3-Clause.
def real_sph_harm(m, n, theta, phi):
    """ Compute real spherical harmonics.

    Where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)     if m > 0
        $Y^0_n$                     if m = 0
        Real($Y^|m|_n$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    See Also
    --------
    scipy.special.sph_harm
    """
    return real_sh_descoteaux_from_index(m, n, theta, phi, legacy=True)
 


# Adapted from dipy.reconst.shm.real_sh_descoteaux_from_index; BSD-3-Clause.
def real_sh_descoteaux_from_index(m, n, theta, phi, legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)      if m > 0
        $Y^0_n$                      if m = 0
        Real($Y^m_n$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        is used for m < 0). Else, implements the basis as defined in
        Descoteaux et al. 2007 (without the absolute value).

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    if legacy:
        # In the case where m < 0, legacy descoteaux basis considers |m|.
        # This basis is used on purpose (the bundled ODFs and all fits use it);
        # DIPY's deprecation warning for it is not relevant here and was removed.
        sh = spherical_harmonics(np.abs(m), n, phi, theta)
    else:
        # In the cited paper, the basis is defined without the absolute value
        sh = spherical_harmonics(m, n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))

    return real_sh
 

# Adapted from dipy.reconst.shm.spherical_harmonics; BSD-3-Clause.
def spherical_harmonics(m, n, theta, phi, use_scipy=True):
    """Compute spherical harmonics.

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.
    use_scipy : bool, optional
        If True, use scipy implementation.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function.

    The usual definitions for ``theta` and `phi`` used in DIPY are interchanged
    in the method definition to agree with the definitions in
    scipy.special.sph_harm, where `theta` represents the azimuthal coordinate
    and `phi` represents the polar coordinate.

    Altough scipy uses a naming convention where ``m`` is the order and ``n``
    is the degree of the SH, the opposite of DIPY's, their definition for both
    parameters is the same as ours, with ``n >= 0`` and ``|m| <= n``.
    """
    if use_scipy:
        return sps.sph_harm_y(n, m, phi, theta).astype(complex)

    x = np.cos(phi)
    val = sps.lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (sps.gammaln(n - m + 1) - sps.gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val


def compute_lm_ordervector(sh_order): 
    '''
    Given a spherical harmonics order in the modified convention (i.e. like in
    Descoteaux et al., 2006), returns the list (array) of the L and M order of
    the coefficients that will be used for the representation in SH.
    '''
    n_c = (sh_order + 1) * (sh_order + 2) // 2
    LM = np.zeros((n_c,2),dtype=int)
    counter = 0
    for kl, l in enumerate(range(0, sh_order + 1, 2)):
        #print 'KL ' + str(l)
        for km, m in enumerate(range(-l, l + 1)):
            #print  'km ' + str(m)
            LM[counter,0] = int(l)
            LM[counter,1] = int(m)
            counter = counter+1
    return LM

def compute_sphmean(coeffs,LM=None):
    '''
    Computes the spherical mean signal given the spherical harmonics coefficients
    and the vector of orders L and M (LM) associated to each coefficients: this
    will be used to locate the coefficient with L=0 and M=0, that is necessary
    to compute the spherical mean.
    '''
    if LM is None:
        return coeffs[0]/np.sqrt(4*np.pi)
    else:
        return coeffs[(LM[:,0]==0) & (LM[:,1]==0)]/np.sqrt(4*np.pi)

def compute_sphvar(coeffs): 
    return np.sum(coeffs[1:]**2)/(4*np.pi)

def l_band_indices(sh_order):
    """Coefficient indices of each even band l = 0, 2, ..., sh_order."""
    LM = compute_lm_ordervector(sh_order)
    return [np.where(LM[:, 0] == l)[0] for l in range(0, sh_order + 1, 2)]

def compute_l_band_power_spectra(coeffs,shorder):
    """Power (sum of squared coefficients) of each even band, as for real data."""
    return np.array([np.sum(coeffs[idx]**2) for idx in l_band_indices(shorder)])

'''
DESIGN and regularization MATRICES
'''

def sh_reconstruction(M, coeffs):
    '''
    Given a SH design matrix M,
    it returns the dot product with the coefficients
    '''
    return np.dot(M,coeffs)

def sh_matrix(sh_order, vecs):
    r""" Compute Spherical Harmonics (SH) matrix M
    Parameters
    ----------
    sh_order : int, even
        Truncation order of the SH.
    vecs : array, Nx3
        array unit directions [X,Y,Z].

    Returns
    --------
    M : array,
        The base of the SH sampled according to vecs.
    """

    r, theta, phi = cart2sphere(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    theta[np.isnan(theta)] = 0

    n_c = (sh_order + 1) * (sh_order + 2) // 2
    shm = real_sph_harm

    M = np.zeros((vecs.shape[0], n_c))
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = shm(m, l, theta, phi)
            counter += 1
    return M

def laplace_beltrami(sh_order):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = (l * (l + 1)) ** 2
            counter += 1

    return np.diag(diagL)

def create_regularization_matrix(shorder,type='lb'):
    if type=='lb':
        R = laplace_beltrami(shorder)
    elif type=='eye':
        R = np.eye(int((shorder+1)*(shorder+2)/2))
    else:
        raise ValueError('either lb or eye are acceptable')
    return R
