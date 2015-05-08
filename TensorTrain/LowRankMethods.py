import numpy as np
import scipy.linalg as scl
import scipy.optimize as sco

import functools as ft

import TensorTrain.LowRank as TLR

def LowRank(eigv,dims,A):
    ''' Computes optimal low-rank decomposition of the full solution for the
    subproblem in ALS.
    
    Parameters:
    --------------
    eigv: pyemma-TICA-object, containing all relevant information about the 
    solution to the full subproblem.
    dims: triple of the dimensions (r_p-1,n,r_p).
    A: ALS-object.
    
    Returns:
    --------------
    LowRank-object, containing all information about the computed low-rank de-
        composition. If None is returned, the optimization failed.
    '''
    # Define a threshold for orthogonality of solutions, remains hard-coded here:
    eps_sol = 1e-14
    # Extract the current eigenvector array:
    Up = eigv.eigenvectors
    # Get the correlation matrices:
    Ctau = eigv.cov_tau
    C0 = eigv.cov
    # Compute initial low-rank decomposition:
    Up = np.reshape(Up,(dims[0]*dims[1],dims[2]*A.M))
    Uk,sk,Wk = scl.svd(Up,full_matrices=False)
    rfull = np.shape(sk)[0]
    Wk = np.dot(np.diag(sk),Wk)
    # Start adaptive rank selection:
    rnew = 1
    while rnew <= rfull:
        # Select only the first rnew singular values:
        U0 = Uk[:,:rnew]
        U1 = Wk[:rnew,:].transpose()
        # Create a low-rank object for this decomposition:
        LR = TLR.LowRank(U0,U1,A.M,Ctau,C0)
        # If orthogonality constraints are not violated, accept:
        if (np.max(np.abs(LR.Orthogonality())) < eps_sol):
            print "Solutions are orthonormal."
            break
        # Otherwise, attempt optimization:
        else:
            print "Attempting Optimization."
            LR = Optimize(LR,A.eps_orth,A.mu0)
            # Check acceptance criteria:
            # First, check if optimization failed entirely:
            if LR == None:
                print "Optimization failed."
                rnew += 1
            else:
                # Get the timescales and check:
                ts = LR.Timescales(A.tau)
                if np.all(ts >= A.eps_rank*A.RefTS()):
                    break
                else:
                    rnew += 1
    print "Rank modified to %d"%rnew
    print ""
    return LR


def Optimize(LR,eps_orth,mu0):
    ''' Runs the constrained optimization problem for each microiteration step
    in ALS. 
     
    Parameters:
    -------------
    LR: LowRank-object, containing all information about the computed low-rank
        decomposition.
    eps_orth: float, tolerance for orthogonality of solutions.
    mu0: float, initial value of penalty parameter.
    
    Returns:
    --------------
    LowRank object, containing all information about the computed low-rank de-
        composition. If None is returned, the optimization failed.
    '''
    # Initialize mu:
    mu = mu0
    # Get initial value of constrained objective:
    L0 = LR.ConstrainedObjective(mu)
    while 1:
        # Get the current components as a vector:
        u0 = LR.GetVector()
        # Set up functions for optimization:
        f = ft.partial(ObjectiveIter,LR=LR,mu=mu)
        fprime = ft.partial(GradientIter,LR=LR,mu=mu)
        # Solve unconstrained problem by conjugate gradient iteration:
        u = sco.fmin_cg(f,u0,fprime=fprime,maxiter=1000,gtol=5e-3,disp=0)
        # Update LR by u:
        LR.SetVector(u)
        # Compute objective function and penalty:
        L = LR.ConstrainedObjective(mu)
        p = LR.Penalty(mu)
        # Check convergence criteria:
        if np.any(np.isnan(u)) or (abs(L)>10*abs(L0)):
            LR = None
            break
        if p < eps_orth:
            break
        else:
            # Update mu:
            mu *= 1.5
    print "Final value of mu: %e"%mu
    print "Final objective function: %.8e"%L
    return LR

def ObjectiveIter(lr,LR,mu):
    ''' This is the function that is called as objective function in the optimi-
    zation problem for the low-rank step.
    
    Parameters:
    --------------
    lr, ndarray, shape(R*(n0+n1),), the current iterate as a vector.
    LR: LowRank object
    mu: float, current value of penalty parameter.
    
    Returns:
    float, the constrained objective function for lr.
    '''
    # Set lr as the component of LR:
    LR.SetVector(lr)
    # Return objective function:
    return LR.ConstrainedObjective(mu)

def GradientIter(lr,LR,mu):
    ''' This is the function that is called in order to retrieve the gradient
    in the optimization problem for the low-rank step.
    
    Parameters:
    --------------
    lr, ndarray, shape(R*(n0+n1),), the current iterate as a vector.
    LR: LowRank object
    mu: float, current value of penalty parameter.
    
    Returns:
    ndarray, shape(R*(n0+n1), the gradient at lr.
    '''
    # Set lr as the component of LR:
    LR.SetVector(lr)
    # Return objective function:
    return LR.Gradient(mu)