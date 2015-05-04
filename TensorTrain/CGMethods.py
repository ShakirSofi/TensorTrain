import numpy as np
import scipy.linalg as scl
import scipy.optimize as sco

from TensorTrain import CanonicalTensor as CT

import functools as ft

def PenaltyCG(TU0,Ctau,C0,J,eps,Ofun0,Lambda0,mu0):
    ''' Runs the constrained optimization problem for the first eigenvalues of
    the projected Transfer-Operator on a low-rank representation. The constraint
    is enforced by penalization, the unconstrained problems are solved by a
    gradient algorithm.
     
    Parameters:
    TU0: CanonicalTensor, the starting value for the tensor.The subspace dimen-
    sion of U0 must be J*n0 and n1, see below.
    Ctau, C0:nd-array, shape(n0*n1,n0*n1), the correlation-matrices of the ba-
    sis functions.
    J: int, the number of eigenfunctions.
    eps: nd-array, shape(0.5*J(J+1),), tolerances for the constraints.
    Ofun0: float, the correct value.
    Lambda0: nd-array, shape(0.5*J(J+1),), initial Lagrange multipliers.
    mu0: float, initial penalty parameter.
     
    Returns:
    U: CanonicalTensor, the optimal low-rank approximation.
    '''
    # Get the subspace dimensions:
    n = np.shape(C0)[0]
    # Get the full representation of U0:
    U0 = TU0.GetVU()
    # Check for valid inputs:
    if not np.shape(U0)[0]==J*n or not np.all(np.shape(C0)==np.shape(Ctau)):
        print "Dimensions do not agree"
        return None
    # Get the size information:
    R = TU0.R
    n0 = TU0.n0
    n1 = TU0.n1 
    # Main Iteration:
    Lambda = Lambda0
    mu = mu0
    U = U0
    TU = TU0
    q = 0
    cst0 = 1e5*np.ones(0.5*J*(J+1))
    cstm = np.zeros((0,0.5*J*(J+1)))
    while 1:
        #print "Iteration %d"%q
        u = TU.GetAll()
        # Set up functions for optimization:
        f = ft.partial(Objective,R=R,n0=n0,n1=n1,Ctau=Ctau,C0=C0,Lambda=Lambda,mu=mu)
        fp = ft.partial(Grad,R=R,n0=n0,n1=n1,Ctau=Ctau,C0=C0,Lambda=Lambda,mu=mu)
        # Solve unconstrained problem:
        u,_,_,_,warnflag = sco.fmin_cg(f,u,fprime=fp,maxiter=1000,gtol=5e-3,full_output=True,disp=0)
        # Get its full representation and reshape it:
        U0 = np.reshape(u[:R*n0],(R,n0))
        U1 = np.reshape(u[R*n0:],(R,n1))
        TU.SetAll(U0,U1)
        U = TU.GetVU()
        U = np.reshape(U,(J,n))
        # Compute Objective function:
        Ofun = Objective(u,R,n0,n1,Ctau,C0,Lambda,mu)
        # Compute the penalty function:
        cst = Constraint(U,C0)
        cstm = np.vstack((cstm,cst))
        #print cst
        Ofun = Ofun - 0.5*mu*np.sum(cst*cst) + np.dot(Lambda,cst)
        #print "Objective Function is %.5e"%Ofun
        if np.any(np.isnan(u)) or (abs(Ofun)>10*abs(Ofun0)) or abs(Ofun)<0.1*abs(Ofun0):
            U0 = np.zeros((R,n0))
            U1 = np.zeros((R,n1))
            TU.SetAll(U0,U1)
            Ofun = 0
            #print "Optimization did not converge."
            break
        if np.all(np.abs(cst)<eps):
            #print "Iteration converged"
            break
        else:
            # Update lambda:
            Lambda = Lambda - mu*cst
            # Check if mu needs to be updated:
            if np.max(np.abs(cst)/cst0) >= 0.95:
                mu = 1.5*mu
            cst0 = np.abs(cst)
            q += 1
    print "Final value of mu: %e"%mu
    print "Final objective function: %.8e"%Ofun
    return (TU,Ofun)

def Objective(u,R,n0,n1,Ctau,C0,Lambda,mu):
    ''' Computes objective function at position u.'''
    # Set up the correct tensor:
    TU = CT.CanonicalTensor(R,n0,n1)
    U0 = np.reshape(u[:R*n0],(R,n0))
    U1 = np.reshape(u[R*n0:],(R,n1))
    TU.SetAll(U0, U1)
    # Get the full representation:
    U = TU.GetVU()
    # Reshape it correctly:
    n = np.shape(C0)[0]
    J = np.shape(U)[0]/n
    UJ = np.reshape(U,(J,n))
    # Get the constraints:
    cst = Constraint(UJ,C0)
    # Compute the unconstrained functional:
    F = -0.5*np.trace(np.dot(UJ,np.dot(Ctau,UJ.transpose())))
    return F + 0.5*mu*np.sum(cst*cst)
    #return F - np.dot(Lambda,cst) + 0.5*mu*np.sum(cst*cst)

def Grad(u,R,n0,n1,Ctau,C0,Lambda,mu):
    ''' Computes gradient at position u+td'''
    # Set up the correct tensor:
    TU = CT.CanonicalTensor(R,n0,n1)
    U0 = np.reshape(u[:R*n0],(R,n0))
    U1 = np.reshape(u[R*n0:],(R,n1))
    TU.SetAll(U0, U1)
    # Compute the gradient:
    L = GetGradient(TU,Lambda,mu,Ctau,C0)
    return L

         
def Constraint(U,C0):
    ''' Evaluates the constraints for a given solution.
     
    Parameters:
    U: ndarray, shape(J,n), the current iterate.
    C0: nd-array, shape(n,n), the time zero correlation matrix.

    Return:
    P: nd-array, shape(0.5*J*(J+1),), values of all the constraint terms.
    '''
    # Initialize:
    J = np.shape(U)[0]
    E = np.eye(J,J)
    P = np.zeros(0.5*J*(J+1))
    # Evaluate all the constraint terms:
    q = 0
    for k in range(J):
        for m in range(k,J):
            P[q] = 0.5*(np.dot(U[k,:],np.dot(C0,U[m,:])) - E[k,m])
            q += 1
    return P

def ApplyBlockMatrix(C,U,k=1,m=0):
    ''' Applies a blockwise replicated version of the matrix C to the vector U.
    
    Parameters:
    C: nd-array, shape(n,n).
    U: nd-array, shape(J,n)
    k,m: int, determine into which blocks the matrix C is replicated. If
    the default is used, it is replicated into all diagonal blocks.
    
    Returns:
    V, nd-array, shape(J*n,), the replicated matrix applied to the vector.    
    '''
    # Get the shape:
    n = np.shape(C)[0]
    J = np.shape(U)[0]
    # Check whether all inputs make sense:
    if not np.all(np.shape(C)==(n,n)):
        print "Matrix dimensions do not agree."
        return None
    if k>J or m>J:
        print "Dimensions do not agree."
        return None
    # Prepare output:
    V = np.zeros((J,n))
    # Evaluate:
    if k <= m:
        V[k,:] += 0.5*np.dot(C,U[m,:])
        V[m,:] += 0.5*np.dot(C,U[k,:])
    else:
        for r in range(J):
            V[r,:] += np.dot(C,U[r,:])
    V = np.reshape(V,(J*n,))
    return V

def GetGradient(TU,Lambda,mu,Ctau,C0):
    ''' Determines the gradient of the unconstrained penalized objective func-
    tion.
     
    Parameters:
    TU: CanonicalTensor, the current iterate.
    Lambda: current value of the penalty parameters.
    Ctau, C0: nd-array, shape(n0*n1,n0*n1), the correlation matrices.
     
    Returns:
    L: nd-array, shape(R(n0+n1),), where R is the rank of the tensor U and n0,
    n1 are the subspace dimensions.
    '''
    # Get the shape parameters:
    n = np.shape(C0)[0]
    # Get a full rep of TU:
    U = TU.GetVU()
    J = np.shape(U)[0]/n
    # Reshape it:
    UJ = np.reshape(U,(J,n))
    # Compute the partial derivative matrix:
    VPU = TU.PartialDerivative()
    # Evaluate the constraints:
    cst = Constraint(UJ,C0)
    # Compute the non-constrained part:
    CU = ApplyBlockMatrix(Ctau,UJ)
    L = -np.dot(CU,VPU)
    # Add the constraint parts:
    q = 0
    for k in range(J):
        for m in range(k,J):
            CU0 = ApplyBlockMatrix(C0,UJ,k,m)
            L -= -mu*cst[q]*np.dot(CU0,VPU)
            #L -= (Lambda[q]- mu*cst[q])*np.dot(CU0,VPU)
            q += 1
    return L

def ApproximateCG(W,C0,ep = 1e-6, Rmax = 0):
    ''' Approximates given matrix W by a low-rank decomposition w.r.t. the norm
    defined through the positive definite matrix C0.
    
    Parameters:
    -------------
    W: ndarray, shape(n0,n1), the matrix to be approximated.
    C0: ndarray, shape(n0*n1,n0*n1), weight matrix defining the scalar product
        w.r.t. which the approximation is carried out.
    ep: float, stopping tolerance for approximation.
    Rmax: int, maximal rank to be tried, defaults to min(n0,n1).
    
    Returns:
    -----------
    wCT: CanonicalTensor object, containing all low-rank variables.
    '''
    # Get the size of input matrix:
    n0, n1 = W.shape[0], W.shape[1]
    # Check the consistency of inputs:
    if not(C0.shape[0] == n0*n1 and C0.shape[1]==n0*n1):
        print "Input dimensions do not agree."
        return
    # Decompose W by SVD:
    UW,sW,VW = scl.svd(W,full_matrices=False)
    # Reshape W to vector:
    W = np.reshape(W,(n0*n1,))
    # Try to approximate with increasing ranks:
    r = 1
    if not Rmax:
        Rmax = np.minimum(n0,n1)
    while r <= Rmax:
        # Use the rank r SVD as initial guess:
        Ur = UW[:,:r]
        Vr = VW[:r,:]
        sr = sW[:r]
        Vr = np.dot(np.diag(sr),Vr)
        # Prepare a canonical tensor object:
        wCT = CT.CanonicalTensor(r,n0,n1)
        wCT.SetAll(Ur.transpose(),Vr)
        wr = wCT.GetAll()
        # Define function objects for optimization:
        f = ft.partial(ObjectiveApprox,R=r,n0=n0,n1=n1,W=W,C0=C0)
        fp = ft.partial(GradApprox,R=r,n0=n0,n1=n1,W=W,C0=C0)
        # Run minimization
        wr,_,_,_,warnflag = sco.fmin_cg(f,wr,fprime=fp,maxiter=1000,gtol=1e-4,full_output=True,disp=0)
        W0 = np.reshape(wr[:r*n0],(r,n0))
        W1 = np.reshape(wr[r*n0:],(r,n1))
        wCT.SetAll(W0,W1)
        # Check convergence:
        if ObjectiveApprox(wr,r,n0,n1,W,C0) < ep and warnflag==0:
            print "Minimization successful for rank r = %d"%r
            break
        elif r == Rmax:
            print "Maximum rank has been reached."
            break
        else:
            r += 1
    if r > Rmax:
        print "Minimization could not be carried out."
        return 0
    else:
        return wCT
            
def ObjectiveApprox(wr,R,n0,n1,W,C0):
    ''' Objective function for approximation of a tensor w.r.t. the norm de-
    fined by C0:
    '''
    # Set up the correct tensor:
    TW = CT.CanonicalTensor(R,n0,n1)
    W0 = np.reshape(wr[:R*n0],(R,n0))
    W1 = np.reshape(wr[R*n0:],(R,n1))
    TW.SetAll(W0,W1)
    # Get the full representation:
    Wr = TW.GetVU()
    # Compute objective function:
    return np.dot(Wr-W,np.dot(C0,Wr-W))

def GradApprox(wr,R,n0,n1,W,C0):
    ''' Gradient for approximation of a tensor w.r.t. the norm defined by C0:
    '''
    # Set up the correct tensor:
    TW = CT.CanonicalTensor(R,n0,n1)
    W0 = np.reshape(wr[:R*n0],(R,n0))
    W1 = np.reshape(wr[R*n0:],(R,n1))
    TW.SetAll(W0, W1)
    # Get the full representation:
    Wr = TW.GetVU()
    # Compute the partial derivatives:
    TWP = TW.PartialDerivative()
    # Compute the full gradient:
    grad = 2*np.dot(np.dot(C0,Wr-W),TWP)
    return grad