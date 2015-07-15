import numpy as np
import scipy.linalg as scl
import scipy.optimize as sco

import functools as ft

import pyemma.util.linalg as pla

def LowRank(eigv,sp,tp,A):
    ''' Performs the actual low-rank decomposition step inside ALS.
    
    Parameters:
    ------------
    eigv: pyemma-TICA object, result of the full four-fold optimization.
    sp,tp: int, the dimensions of the two product bases that were used, needed
    for the reshape of the solution.
    A: ALS-object
    
    Returns:
    ------------
    Up: ndarray, shape(r_p-1*n,R), where R is the new rank.
    '''
    # Extract the correlation matrices and the full solution:
    Upp = eigv.eigenvectors
    C0 = eigv.cov
    Ctau = eigv.cov_tau
    # Extract objective function from full computation:
    Lref = -np.sum(eigv.eigenvalues[:A.M])
    # Update the reference value in A:
    A.UpdateLref(Lref)
    # Reshape Upp and perform SVD:
    Upp = np.reshape(Upp,(sp,tp*A.M))
    V,_,_ = scl.svd(Upp)
    # Try to optimize with increasing ranks:
    r = 1
    count = 0
    rmax = np.minimum(A.rmax,sp)
    while r <= rmax:
        print "Attempting rank %d."%r
        # If the rank equals the dimension of the left space, use identity:
        if r == sp:
            Up = np.eye(sp,sp)
            L = Lref
            break
        # Initialize optimization by first columns of V:
        Up = (V[:,1:r]).copy()
        print Up.shape
        # Perform optimization:
        Up,L = Optimize(Up,Ctau,C0,sp,tp,r,A.M)
        print "Optimization finished."
        # Check if the result is good enough:
        if L < (A.tol*A.Lref):
            A.retries.append(count+1)
            print "Result accepted. Rank set to %d"%r
            break
        else:
            if (count <= A.cmax) and (r > 1):
                print "Result insufficient. Retrying."
                count += 1
            else:
                print "Result insufficient. Increasing rank."
                r += 1
                count = 0
        if r > rmax:
            print "Optimization failed, maximum rank reached."
            Up = None
    return (Up,L)

def Optimize(Up,Ctau,C0,sp,tp,R,M):
    # Reshape Up:
    u0 = np.reshape(Up,(sp*(R-1),))
    # Reshape Ctau, C0:
    Ctau = np.reshape(Ctau,(sp,tp,sp,tp))
    C0 = np.reshape(C0,(sp,tp,sp,tp))
    if R > 1:
        # Define objective function:
        f = ft.partial(Objective,Ctau=Ctau,C0=C0,sp=sp,tp=tp,R=R,M=M)
        # Optimize:
        res = sco.minimize(f,u0,method="Anneal")
        # Extract result and objective function:
        u = res.x
        L = res.fun
        # Reshape u and return:
        u = np.reshape(u,(sp,R-1))
        # Normalize Up:
        u = Normalize(u,C0)
    else:
        L = Objective(u0,Ctau,C0,sp,tp,R,M)
        u = np.reshape(u0,(sp,R-1))
    # Finally, add the column for the constant:
    u = np.hstack((np.eye(sp,1),u))
    return (u,L)   
    

def Objective(u,Ctau,C0,sp,tp,R,M):
    ''' This is the actual objective function of the low-rank optimization
    problem.
    
    Parameters:
    -------------
    u: Current estimate.
    Ctau, C0: The correlation matrices.
    '''
    # Reshape u:
    U = np.reshape(u,(sp,R-1))
    # Add a column encoding the constant:
    U = np.hstack((np.eye(sp,1),U))
    # Compute the two correlation matrices:
    Ctau = np.einsum('ij,iklm,ln->jknm',U,Ctau,U)
    C0 = np.einsum('ij,iklm,ln->jknm',U,C0,U)
    # Reshape the correlation matrices:
    Ctau = np.reshape(Ctau,(R*tp,R*tp))
    C0 = np.reshape(C0,(R*tp,R*tp))
    # Solve the optimization problem:
    D,_ = pla.eig_corr(C0,Ctau)
    if (D is None) or (D.shape[0] < M):
        L = 0
    else:
        L = -np.sum(D[:M])
    return L
    
def Normalize(Up,C0):
    ''' Normalize optimization result such that the corresponding basis
    have unit length.
    '''
    # Compute C0 matrix:
    C0 = np.einsum('ij,iklm,ln->jknm',Up,C0,Up)
    # Compute the average norms of the basis functions involving each column of
    # Up. First extract the diagonal norms:
    avg_norms = np.einsum('ijij->ij',C0) 
    # Get the averages:
    avg_norms = np.sqrt(np.mean(avg_norms,axis=1))
    # Divide the columns of Up:
    Up /= avg_norms
    return Up