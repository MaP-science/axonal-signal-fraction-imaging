import numpy as np
import cvxpy
from .posoptimize import PositiveDefiniteLeastSquares, load_sdp_constraints
from . import sphericalharmonics as sh
# from diffusion_methods import sh2 as sh
from joblib import Parallel, delayed
from . import timemanagement as tm
import warnings

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

def fit_sh_relaxed_positivity(shell, M, R,lambda_reg,M_positivity,verbose=False,solver=None):
    '''
    shell: data points on a spherical shell (possibly with equidistant sampling)
    M:     SH design matrix
    R:     regularization matrix (e.g. Laplace-beltrami)
    
    implements "relaxed" positivity constraints contained
    within a M_positivity matrix that has NOT been pre-multiplied
    by the regularization value lambda_reg
    '''

    c = cvxpy.Variable(M.shape[1])
    design_matrix = cvxpy.Constant(M)

    objective = cvxpy.Minimize(
        cvxpy.sum_squares(design_matrix @ c - shell) +
        lambda_reg * cvxpy.quad_form(c, R)
    )
    
    constraints = [(M_positivity @ c) >=0]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(verbose=verbose,solver=solver)
    coef = np.asarray(c.value).squeeze()

    return coef


def preload_sdp(sh_order,M):
    sdp_constraints = load_sdp_constraints('real_sh_descoteaux',sh_order)
    sdp = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
    return sdp 

def fit_sh_necessary_positivity(shell, M,sdp_constraints,sdppreloaded=None,verbose=False,solver=None):
    '''
    Solves the SH fitting problem by imposing NECESSARY constraints for positivity
    as proposed by Dela Haije et al., 2020.
    
    You can pass a full sdppreloaded as
        sdp_constraints = posoptimize.load_sdp_constraints('real_sh_descoteaux',sh_order)
        sdppreloaded = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
    OR with
        sdppreloaded = sphericalharmonicsfit.preload_sdp(sh_order,M)
        
    Note: if the chosen solver is 'MOSEK'
    Make sure to have a MOSEK License if you do not want to have license errors
    (which are difficult to debug when using parallelization)
    Make also sure that you scale the input data (multiply by a big number) to avoid numerical issues
    '''
    if sdppreloaded is None:
        sdp = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
    else:
        sdp = sdppreloaded
    
    coef = sdp.solve(M, shell,solver=solver,verbose=verbose)
     
    return coef


def create_regularization_matrix(shorder,type='lb'):
    if type=='lb':
        R = sh.laplace_beltrami(shorder)
    elif type=='eye':
        R = np.eye(int((shorder+1)*(shorder+2)/2))
    else:
        raise ValueError('either lb or eye are acceptable')
    return R





def worker_fit_sh(tuple_item):
    start = tm.get_time()
    signal          = tuple_item[0]
    M               = tuple_item[1]
    R               = tuple_item[2]
    lambda_reg      = tuple_item[3]
    
    tot           = tuple_item[4]
    x             = tuple_item[5]
    n_workers     = tuple_item[6]
    progressevery = tuple_item[7]

       
    coeffs = fit_sh(signal, M, R,lambda_reg)
    #recon = sh.sh_reconstruction(M, coeffs)
    
    
    strt = tm.print_expected_time(start,x,tot,every_integer=progressevery,n_workers=n_workers)

    return tuple(coeffs)

def worker_fit_sh_relaxed_positivity(tuple_item): # USED
    start = tm.get_time()
    signal          = tuple_item[0]
    M               = tuple_item[1]
    M_positivity    = tuple_item[2]
    R               = tuple_item[3]
    lambda_reg      = tuple_item[4]
    cvxpyverbose    = tuple_item[5]
    solver          = tuple_item[6]

    tot           = tuple_item[7]
    x             = tuple_item[8]
    n_workers     = tuple_item[9]
    progressevery = tuple_item[10]

    
    
    try:
        coeffs =  fit_sh_relaxed_positivity(signal, M, R,lambda_reg,M_positivity,cvxpyverbose,solver)
        if coeffs is None:
            msg = 'worker '+ str(x) +' relaxed positivity failed. Solving REGULARIZED least-squares instead..'
            warnings.warn(msg)
            coeffs = fit_sh(signal, M, R,lambda_reg=lambda_reg)
    except:
        msg = 'worker '+ str(x) +' relaxed positivity failed. Solving REGULARIZED least-squares instead..'
        warnings.warn(msg)
        
        if solver=='MOSEK':
            msg2 = 'You are using MOSEK: try to scale input data by a large number, e.g. 10000,\nand dont forget to then divide teh output by the same:\nodfcoeff=odfcoeff/10000'
            warnings.warn(msg2) 
            
        
        coeffs = fit_sh(signal, M, R,lambda_reg=lambda_reg)
        # print('coeffs',coeffs)
        # coeffs = np.ones(M.shape[1])
        # print(signal.shape,M.shape,R.shape,lambda_reg)
    #recon = sh.sh_reconstruction(M, coeffs)
    
    
    strt = tm.print_expected_time(start,x,tot,every_integer=progressevery,n_workers=n_workers)

    # Dealing with Capricious MOSEK
    try:
        result = tuple(coeffs)
    except:
        msg = 'worker '+ str(x) +' falied'
        warnings.warn(msg)
        coeffs = np.zeros(M.shape[1])
        result = tuple(coeffs)

    return result


def worker_fit_sh_necessary_positivity(tuple_item):
    start = tm.get_time()
    signal          = tuple_item[0]
    M               = tuple_item[1]
    sdp_constraints = tuple_item[2]
    sdppreloaded    = tuple_item[3]
    cvxpyverbose    = tuple_item[4]
    solver          = tuple_item[5]
    
    tot           = tuple_item[6]
    x             = tuple_item[7]
    n_workers     = tuple_item[8]
    progressevery = tuple_item[9]

    
    # if sdppreloaded is None:
    #         sdppreloaded = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
    # # coeffs = fit_sh_necessary_positivity(signal, M, sdp_constraints,sdppreloaded,cvxpyverbose)
    # coeffs = fit_sh_necessary_positivity(signal, M, sdp_constraints,None,cvxpyverbose,solver)
    
    try:
        if sdppreloaded is None:
            sdppreloaded = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
        coeffs = fit_sh_necessary_positivity(signal, M, sdp_constraints,sdppreloaded,cvxpyverbose)
    except:
        msg = 'worker '+ str(x) +' necessary positivity failed. Solving NON-regularized least-squares instead'
        warnings.warn(msg)
        R = np.zeros((M.shape[1],M.shape[1]))
        coeffs =  fit_sh(signal, M, R,lambda_reg=0)
    # recon = sh.sh_reconstruction(M, coeffs)
    
    
    strt = tm.print_expected_time(start,x,tot,every_integer=progressevery,n_workers=n_workers)

    return tuple(coeffs)


def fit_sh_all_data(data_rcs,dirs_xyz,sh_order,n_workers,mask,type='ls',regularization='lb',lambda_reg=1e-001,progressevery=100,cvxpyverbose=False,solver='MOSEK',inputscale=10000):
    '''
    type = 'ls' OR 'relaxedpos' OR 'necessarypos'
    regularization = 'lb' OR 'eye'
    inputscale scales the input for preventing numerical errors (Particularly useful when using MOSEK solver)
    
    for the possible solvers check cvxpy.installed_solvers()
    '''
    
    if (type!='ls') & (type!='relaxedpos') & (type!='necessarypos'):
        raise ValueError('Incorrect type specified')
    
    use_input_scale = False
    if (inputscale!=1)&(type!='ls'):
        use_input_scale = True
        if inputscale<=0:
            raise ValueError('inputscale must be a non-zero positive scalar value')
        print('Using inputscale=',inputscale,', make sure that input data is not too small nor too large!')
    else:
        warnings.warn('It may be safer to use inputscale != 1 for numerical stability by making sure that input data is not too small nor too large!!')
    
    if use_input_scale:
        data_rcs = data_rcs*inputscale
    
    if (type!='ls'):
        if solver not in cvxpy.installed_solvers():
            solver = None
            warnings.warn('solver was set to None cause selected '+ solver + ' is not installed!')
    
    if (regularization!='lb') & (type!='eye'):
        raise ValueError('Incorrect regularization specified')
    
    progressevery = int(progressevery)
    
    nR,nC,nS,ndirs = data_rcs.shape
    # data_sphmean = np.zeros((nR,nC,nS))
    # data_sphvar = np.zeros((nR,nC,nS))

    # LM = sh.compute_lm_ordervector(sh_order)
    M = sh.sh_matrix(sh_order, dirs_xyz)
    
    
    if (type=='ls') | (type=='relaxedpos'):
        R = create_regularization_matrix(sh_order,type=regularization)
    
    if type=='relaxedpos':
        M_positivity = sh.sh_matrix(sh_order, sh._directions724)
    
    if type=='necessarypos':
        sdp_constraints = load_sdp_constraints('real_sh_descoteaux',sh_order)
        sdppreloaded = PositiveDefiniteLeastSquares(M.shape[1],A=sdp_constraints)
    
    
    data_sphcoeff = np.zeros((nR,nC,nS,M.shape[1]))

    list_data_idxs = []
    tot = np.sum(mask.astype(int))
    cnt = 0
    for r in range(nR):
        for c in range(nC):
            for s in range(nS):
                if mask[r,c,s]==1:
                    if type=='ls':
                        # list_data_idxs.append((data_rcs[r,c,s,:],M,R,lambda_reg,LM,tot,cnt,n_workers,progressevery))
                        list_data_idxs.append((data_rcs[r,c,s,:],M,R,lambda_reg,tot,cnt,n_workers,progressevery))
                    elif type=='relaxedpos':
                        # list_data_idxs.append((data_rcs[r,c,s,:],M,M_positivity,R,lambda_reg,LM,cvxpyverbose,tot,cnt,n_workers,progressevery))
                        list_data_idxs.append((data_rcs[r,c,s,:],M,M_positivity,R,lambda_reg,cvxpyverbose,solver,tot,cnt,n_workers,progressevery))
                    elif type=='necessarypos':
                        # list_data_idxs.append((data_rcs[r,c,s,:],M,sdp_constraints,sdppreloaded,LM,cvxpyverbose,tot,cnt,n_workers,progressevery))
                        list_data_idxs.append((data_rcs[r,c,s,:],M,sdp_constraints,sdppreloaded,cvxpyverbose,solver,tot,cnt,n_workers,progressevery))
                    else:
                        raise ValueError('Selected type is not implemented')
                    cnt = cnt + 1

    
    if type=='ls':
        worker_fun = worker_fit_sh
        print('-> Include',regularization,'regularization with lambda',lambda_reg)
    elif type=='relaxedpos':
        worker_fun = worker_fit_sh_relaxed_positivity
        print('-> Include',regularization,'regularization with lambda',lambda_reg,'and relaxed positivity constraints')
    elif type=='necessarypos':
        print('-> Include NO regularization and NECESSARY positivity constraints')
        worker_fun = worker_fit_sh_necessary_positivity
    else:
        raise ValueError('Selected type is not implemented')
    
    results = Parallel(n_jobs=n_workers,verbose=1)(delayed(worker_fun)(list_data_idxs[x]) for x in range(cnt))
    
    cnt=0
    for r in range(nR):
        for c in range(nC):
            for s in range(nS):
                if mask[r,c,s]==1:
                    fit_result = results[cnt]
                    cnt = cnt + 1
                    data_sphcoeff[r,c,s,:] = fit_result

    if use_input_scale:
        data_rcs = data_rcs/inputscale
        data_sphcoeff = data_sphcoeff/inputscale
    
    return data_sphcoeff

        


def compute_l_band_power_spectra_all_data(coeffs,shorder):


    n_coeffs = coeffs.shape[-1]
    n_spectra = int(shorder/2+1)
    
    lbps = np.zeros(coeffs.shape[0:-1]+tuple([n_spectra]))

    for i in range(n_spectra):
        lbps[...,i] = np.sum(coeffs[...,sh.shidxs[i]]**2,axis=-1)

    return lbps