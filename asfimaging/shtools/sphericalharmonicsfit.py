import logging
import warnings

import numpy as np
import cvxpy
from joblib import Parallel, delayed

from . import sphericalharmonics as sh

logger = logging.getLogger(__name__)

'''
FITTING SPHERICAL HARMONICS COEFFICIENTS
'''

def fit_sh(shell, M, R,lambda_reg=0):
    '''
    shell: data points on a spherical shell (possibly with equidistant sampling)
    M:     SH design matrix
    R:     regularization matrix (e.g. Laplace-beltrami)
    It returns the spherical harmonics coefficients
    '''
    
    if np.all(shell == 0):
        coef = np.zeros(M.shape[1])
    else:
        pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambda_reg*R), M.T)
        coef = np.dot(pseudoInv, shell)
    return coef

_POSITIVITY_PROBLEMS = {}


def _positivity_problem(M, R, lambda_reg, M_positivity):
    """Build (once per process) a parameterised CVXPY problem for this design.

    The signal is a CVXPY Parameter, so repeated fits with the same directions
    reuse the compiled problem instead of re-canonicalising it every voxel.
    """
    key = (M.shape, M.tobytes(), M_positivity.shape, float(lambda_reg),
           R.tobytes() if lambda_reg else b'')
    cached = _POSITIVITY_PROBLEMS.get(key)
    if cached is None:
        c = cvxpy.Variable(M.shape[1])
        shell = cvxpy.Parameter(M.shape[0])
        cost = cvxpy.sum_squares(M @ c - shell)
        if lambda_reg:
            cost = cost + lambda_reg * cvxpy.quad_form(c, cvxpy.psd_wrap(R))
        problem = cvxpy.Problem(cvxpy.Minimize(cost), [(M_positivity @ c) >= 0])
        cached = (problem, c, shell)
        _POSITIVITY_PROBLEMS[key] = cached
    return cached


def fit_sh_relaxed_positivity(shell, M, R,lambda_reg,M_positivity,verbose=False,solver=None):
    '''
    shell: data points on a spherical shell (possibly with equidistant sampling)
    M:     SH design matrix
    R:     regularization matrix (e.g. Laplace-beltrami); ignored when lambda_reg == 0

    implements "relaxed" positivity constraints contained
    within a M_positivity matrix that has NOT been pre-multiplied
    by the regularization value lambda_reg
    '''
    prob, c, shell_param = _positivity_problem(M, R, lambda_reg, M_positivity)
    shell_param.value = np.asarray(shell, dtype=float).reshape(-1)
    # No warm start: a reused (cached) problem must not depend on the previous fit.
    prob.solve(verbose=verbose, solver=solver, warm_start=False)
    if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE) or c.value is None:
        raise RuntimeError(f"Constrained SH fitting failed: {prob.status}")
    coef = np.asarray(c.value, dtype=float).reshape(-1)
    if coef.size != M.shape[1] or not np.all(np.isfinite(coef)):
        raise RuntimeError("Constrained SH fitting returned invalid coefficients")

    return coef


def fit_sh_single(shell, M, R, lambda_reg=0.0, positivity=True, M_positivity=None,
                  solver='SCS', verbose=False, label=''):
    """Fit one shell, used identically for real voxels and synthetic signals.

    With positivity, a failed constrained fit falls back (with a warning) to
    least squares using the same lambda_reg.
    """
    if not positivity:
        return fit_sh(shell, M, R, lambda_reg=lambda_reg)
    if np.all(np.asarray(shell) == 0):
        return np.zeros(M.shape[1])
    try:
        return fit_sh_relaxed_positivity(shell, M, R, lambda_reg, M_positivity, verbose, solver)
    except (cvxpy.error.SolverError, RuntimeError) as exc:
        warnings.warn(
            f"{label}constrained SH fitting failed ({exc}); "
            "using least squares without positivity constraints.",
            RuntimeWarning,
        )
        return fit_sh(shell, M, R, lambda_reg=lambda_reg)


def worker_fit_sh(signal, M, R, lambda_reg):
    return fit_sh(signal, M, R, lambda_reg)


def worker_fit_sh_relaxed_positivity(signal, M, M_positivity, R, lambda_reg, cvxpyverbose, solver, x):
    return fit_sh_single(signal, M, R, lambda_reg=lambda_reg, positivity=True,
                         M_positivity=M_positivity, solver=solver,
                         verbose=cvxpyverbose, label=f"Voxel {x}: ")


def fit_sh_all_data(data_rcs,dirs_xyz,sh_order,n_workers,mask,type='ls',regularization='lb',lambda_reg=0.0,cvxpyverbose=False,solver='SCS',inputscale=10000):
    '''
    type = 'ls' OR 'relaxedpos'
    regularization = 'lb' OR 'eye'
    inputscale scales constrained-fit inputs for numerical stability.

    for the possible solvers check cvxpy.installed_solvers()
    Progress is reported by joblib.
    '''

    if type not in ('ls', 'relaxedpos'):
        raise ValueError("Unsupported fitting type. Use 'ls' or 'relaxedpos'.")

    if regularization not in ('lb', 'eye'):
        raise ValueError("regularization must be 'lb' or 'eye'")
    if sh_order < 0 or sh_order % 2:
        raise ValueError("SH order must be a non-negative even integer")
    if not np.isfinite(inputscale) or inputscale <= 0:
        raise ValueError("inputscale must be finite and positive")
    if n_workers == 0:
        raise ValueError("n_workers must not be zero")
    data_rcs = np.asarray(data_rcs, dtype=float)
    mask = np.asarray(mask) > 0
    if data_rcs.ndim != 4 or mask.shape != data_rcs.shape[:3]:
        raise ValueError("Expected 4D diffusion data and a matching 3D mask")
    if np.shape(dirs_xyz) != (data_rcs.shape[-1], 3):
        raise ValueError("Directions must have shape (number of volumes, 3)")
    use_input_scale = inputscale != 1 and type != 'ls'
    if use_input_scale:
        data_rcs = data_rcs * inputscale
    if type != 'ls' and solver is not None and solver not in cvxpy.installed_solvers():
        warnings.warn(
            f"Solver {solver!r} is not installed; allowing CVXPY to choose a solver.",
            RuntimeWarning,
        )
        solver = None

    M = sh.sh_matrix(sh_order, dirs_xyz)
    R = sh.create_regularization_matrix(sh_order,type=regularization)

    voxels = data_rcs[mask]  # (n_voxels, n_dirs), in r, c, s order
    if type=='ls':
        logger.info("SH fit: least squares, %s regularization, lambda %g", regularization, lambda_reg)
        tasks = (delayed(worker_fit_sh)(signal, M, R, lambda_reg) for signal in voxels)
    else:
        logger.info("SH fit: relaxed positivity, %s regularization, lambda %g", regularization, lambda_reg)
        M_positivity = sh.sh_matrix(sh_order, sh._positivity_directions)
        tasks = (delayed(worker_fit_sh_relaxed_positivity)(signal, M, M_positivity, R, lambda_reg,
                                                           cvxpyverbose, solver, x)
                 for x, signal in enumerate(voxels))

    results = Parallel(n_jobs=n_workers,verbose=1)(tasks)

    data_sphcoeff = np.zeros(data_rcs.shape[:3] + (M.shape[1],))
    if len(results):
        data_sphcoeff[mask] = np.asarray(results).reshape(len(results), M.shape[1])

    if use_input_scale:
        data_sphcoeff = data_sphcoeff/inputscale

    return data_sphcoeff
