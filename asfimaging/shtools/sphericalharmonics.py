import numpy as np
import scipy.special as sps
from warnings import warn

_directions724 = np.load(r'/work3/s165541/workspace/WP1/MLtools/shtools/data/directions724.npy')

# for l in np.unique(LM[:,0]):
#     idxs = LM[:,0]==l
#     idxs_num = [i for i, x in enumerate(idxs) if x]
#     print(idxs_num)

shidxs = [[0],
[1, 2, 3, 4, 5],
[6, 7, 8, 9, 10, 11, 12, 13, 14],
[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
[28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
[45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
[66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]]


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
        # In the case where m < 0, legacy descoteaux basis considers |m|
        warn('The legacy descoteaux07 SH basis is outdated and will be '
             'deprecated in a future DIPY release. Consider using the new '
             'descoteaux07 basis.', category=PendingDeprecationWarning)
        sh = spherical_harmonics(np.abs(m), n, phi, theta)
    else:
        # In the cited paper, the basis is defined without the absolute value
        sh = spherical_harmonics(m, n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))

    return real_sh
 

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

def compute_l_band_power_spectra(coeffs,shorder): 
    n_spectra = int(shorder/2+1)
    lbps = np.zeros(n_spectra)
    for i in range(n_spectra):
        lbps[i] = np.sum(coeffs[shidxs[i]]**2)
    return lbps

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

def extented_sh_matrix(sh_order_list,list_of_bvecs):
    '''
    Given a list of spherical harmonics orders and a list of bvectors (e.g. one set)
    of b-vectors for each b-value, then it creates an EXTENDED SH matrix
    '''
    
    n_rows = 0
    sh_order_max = int(np.max(np.array(sh_order_list)))
    for vecs in list_of_bvecs:
        n_rows = n_rows + vecs.shape[0]
    n_cols =  ((sh_order_max + 1) * (sh_order_max + 2) // 2)
    
    sh_extended_matrix = np.zeros((n_rows,n_cols))
    L_order_extended_matrix = np.zeros((n_rows,n_cols))*np.nan
    previous_row_number = 0
    num_shells = len(list_of_bvecs)
    for k,vecs in enumerate(list_of_bvecs):
        sh_order = sh_order_list[k]
        shmtrx = sh_matrix(sh_order, vecs)
        Lmatrix = np.ones_like(shmtrx)
        LM = compute_lm_ordervector(sh_order)
        Lmatrix = Lmatrix*LM[:,0]
        shmtrx_nrows,shmtrx_ncols = shmtrx.shape
        sh_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols] = shmtrx#shmtrx[:,::-1][:,0:-1]
        L_order_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols] = Lmatrix
        previous_row_number = previous_row_number+shmtrx_nrows
    
    L_order_indexs = []
    lower_limit = 0#list_of_bvecs[0].shape[0]
    
    for k,vecs in enumerate(list_of_bvecs):
        n_rows_block = vecs.shape[0]
        upper_limit = lower_limit+n_rows_block
        #print(lower_limit,upper_limit)
        L_order_indexs_block = []

        sh_order = sh_order_list[k]
        LM = compute_lm_ordervector(sh_order)
        for l in np.unique(LM[:,0]):
            if l>=0:
                #print(l)
                whereres = np.where(L_order_extended_matrix==l)
                idxs = (whereres[0] < upper_limit) & (whereres[0] >= lower_limit)
                rows = whereres[0][idxs]
                cols = whereres[1][idxs]
                L_order_indexs_block.append(tuple((rows,cols)))
        lower_limit = upper_limit

        L_order_indexs.append(L_order_indexs_block)

    return sh_extended_matrix,L_order_extended_matrix,L_order_indexs

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