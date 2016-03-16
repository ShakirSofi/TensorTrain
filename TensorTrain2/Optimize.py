import numpy as np
import scipy.optimize as sco

import functools as ft

import variational.solvers.direct as vsd

def LowRank(eigv, rpm, n, tp, A):
    ''' Performs the actual low-rank decomposition step inside ALS.
    
    Parameters:
    ------------
    eigv: result of the full four-fold optimization.
    sp,tp: int, the dimensions of the two product bases that were used, needed
    for the reshape of the solution.
    A: ALS-object
    
    Returns:
    ------------
    Up: ndarray, shape(r_p-1*n,R), where R is the new rank.
    '''
    # Extract the correlation matrices and the full solution:
    Upp = eigv.V
    C0 = eigv.C0
    Ctau = eigv.Ct
    # Compute the dimension of the first double-product basis:
    sp = rpm * n
    # Extract objective function from full computation:
    Lref = -np.sum(eigv.d[:A.M])
    # Update the reference value in A:
    A.UpdateLref(Lref)
    # Reshape Upp and perform SVD:
    #Upp = np.reshape(Upp, (sp, tp*A.M))
    V = np.eye(sp, sp)
    for i in range(1, rpm):
        cx = V[:, i].copy()
        V[:, i] = V[:, i*n].copy()
        V[:, i*n] = cx
    # Try to optimize with increasing ranks:
    r = 1
    rmax = np.minimum(A.rmax,sp)
    while r <= rmax:
        print "Attempting rank %d."%r
        # If the rank equals the dimension of the left space, use identity:
        if r == sp:
            Up = np.eye(sp, sp)
            L = Lref
            break
        # Initialize optimization by first columns of V:
        Up = (V[:, 1:r]).copy()
        # Perform optimization:
        Up,L = Optimize(Up, Ctau, C0, sp, tp, r, A.M, A.gtol)
        print "Optimization finished."
        # Check if the result is good enough:
        if L < (A.tol*A.Lref):
            print "Result accepted. Rank set to %d"%r
            break
        else:
            print "Result insufficient. Increasing rank."
            print "Objective needed: %.5f"%(A.tol*A.Lref)
            print "Result: %.5f"%L
            r += 1
        if r > rmax:
            print "Optimization failed, maximum rank reached."
            Up = None
    return (Up,L)

def Optimize(Up,Ctau,C0,sp,tp,R,M,gtol):
    # Reshape Up:
    u0 = np.reshape(Up, (sp*(R-1),))
    # Reshape Ctau, C0:
    Ctau = np.reshape(Ctau, (sp, tp, sp, tp))
    C0 = np.reshape(C0, (sp, tp, sp, tp))
    if R > 1:
        # Define objective function:
        f = ft.partial(Objective, Ctau=Ctau.copy(), C0=C0.copy(), sp=sp, tp=tp, R=R, M=M)
        # Optimize:
        res = sco.minimize(f, u0.copy(), method="CG", jac=True, tol=gtol)
        # Extract result and objective function:
        u = res.x
        L = res.fun
        # Reshape u and return:
        u = np.reshape(u, (sp, R-1))
        # Normalize Up:
        u = Normalize(u, C0)
    else:
        L = Objective(u0, Ctau, C0, sp, tp, R, M, return_grad=False)
        u = np.reshape(u0, (sp, R-1))
    # Finally, add the column for the constant:
    u = np.hstack((np.eye(sp, 1), u))
    return (u, L)
    

def Objective(u, Ctau, C0, sp, tp, R, M, return_grad=True):
    ''' This is the actual objective function of the low-rank optimization
    problem.
    
    Parameters:
    -------------
    u: Current estimate.
    Ctau, C0: The correlation matrices.
    '''
    # Reshape u:
    U = np.reshape(u, (sp, R-1))
    # Add a column encoding the constant:
    U = np.hstack((np.eye(sp, 1), U))
    # Compute the two correlation matrices:
    Ctaup = np.einsum('ij,iklm,ln->jknm', U, Ctau, U)
    C0p = np.einsum('ij,iklm,ln->jknm', U, C0, U)
    # Reshape the correlation matrices:
    Ctaup = np.reshape(Ctaup, (R*tp, R*tp))
    C0p = np.reshape(C0p, (R*tp, R*tp))
    # Solve the optimization problem:
    D,X = vsd.eig_corr_qr(C0p.copy(), Ctaup.copy())
    # Check for failure of the problem:
    if (D is None) or (D.shape[0] < M):
        if D is None:
            print "Warning: Problem could not be solved at all."
        else:
            print "Warning: Only %d eigenvalues could be computed."%D.shape[0]
        L = 0
        if R == 1 or return_grad == False:
            return L
        else:
            grad = None
            return (L, grad)
    else:
        # Compute the objective function:
        L = -np.sum(D[:M])
        if R == 1 or return_grad == False:
            return L
        else:
            # Restrict the eigenvectors to necessary number:
            X = X[:, :M]
            # Compute the Jacobian of Ctau.C0 w.r.t. U:
            J = Jacobian(Ctau, C0, U)
            # Compute the gradient w.r.t. Ctau,C0:
            g = GradCMatrix(X, D, R, tp)
            # Compute the final gradient by dot-product:
            grad = np.dot(g, J)
            grad = grad.flatten()
            return (L, grad, Ctaup, C0p)

def GradCMatrix(X,D,R,tp):
    '''
    Compute the gradient of the full objective function w.r.t. the entries of Ctau and C0
    '''
    # Get the number of eigenvectors:
    M = X.shape[1]
    # Prepare output:
    g1 = np.zeros((R*tp, R*tp))
    g2 = np.zeros((R*tp, R*tp))
    # Iteratively compute the entries;
    for m in range(M):
        # Update the Ctau-part:
        g1 -= np.outer(X[:, m], X[:, m])*(2 - np.eye(R*tp, R*tp))
        # Update the C0-part:
        g2 += D[m]*np.outer(X[:, m], X[:, m])*(2 - np.eye(R*tp, R*tp))
    # Extract upper triangles and glue together:
    iu = np.triu_indices(g1.shape[0])
    g = np.hstack((g1[iu], g2[iu]))
    g = g[None, :]
    return g
    
def Jacobian(Ctau,C0,U):
    ''' Computes the Jacobian of the correlation matrices Ctau,C0 w.r.t. the
    low-rank solution U:'''
    # Get the shapes:
    sp,R = U.shape
    tp = Ctau.shape[1]
    # Compute the two summands:
    A1 = np.kron(U, np.eye(R,R)[:, 1:])
    A1 = np.reshape(A1, (sp, R, R, R-1))
    A2 = np.transpose(A1, [0, 2, 1, 3])
    # Compute the Jacobian by einsum:
    J1 = np.einsum('ijkl,kmno->mjnlio', Ctau, A1) + np.einsum('ijkl,imno->mjnlko',Ctau, A2)
    J2 = np.einsum('ijkl,kmno->mjnlio', C0, A1) + np.einsum('ijkl,imno->mjnlko', C0, A2)
    # Reshape, extract the upper triangle:
    iu = np.triu_indices(R*tp)
    J1 = np.reshape(J1,(R*tp, R*tp, sp*(R-1)))
    J1 = J1[iu[0],iu[1],:]
    J2 = np.reshape(J2, (R*tp, R*tp, sp*(R-1)))
    J2 = J2[iu[0], iu[1], :]
    # Glue them together:
    J = np.vstack((J1, J2))
    return J
        

def Normalize(Up,C0):
    ''' Normalize optimization result such that the corresponding basis
    have unit length.
    '''
    # Compute C0 matrix of left basis:
    C0 = C0[:, 0, :, 0]
    # Compute the square norms of the interfaces:
    inorms = np.dot(Up.transpose(), np.dot(C0, Up))
    inorms = np.diag(inorms) 
    # Divide the columns of Up:
    Up /= np.sqrt(inorms)
    return Up