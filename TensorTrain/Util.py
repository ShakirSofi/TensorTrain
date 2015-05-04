import numpy as np
import scipy.linalg as scl

import pyemma.coordinates as pco

def ApplyLinearTransform(Y,U,filename):
    ''' Apply linear transformation U to time-series given by Y.
    
    Parameters:
    -------------
    Y, pyemma-reader, containing time series of basis functions.
    U, ndarray, shape (r,s), where r must be identical to dimension of Y and s
        is the number of linear combinations to be extracted.
    filename: str, name to be used to save the data for the new time series.
    
    Returns:
    -------------
    pyemma-reader, containing the time-series of all the linear transform 
        applied to Y.'''
    # Get the dimension of the new time-series:
    r = U.shape[1]
    # Get the iterator for the time-series:
    I = Y.iterator()
    # Prepare an empty array for the trajectory pieces:
    file_names = []
    q = 0
    ieval = np.zeros((0,r))
    # Compute the products chunk by chunk:
    for piece in I:
        # Get the trajectory number and the data:
        traj_id = piece[0]
        piece = piece[1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "%d.npy"%q,ieval)
            file_names.append(filename + "%d.npy"%q)
            ieval = np.zeros((0,r))
            q += 1
        # Apply linear transform:
        piece = np.dot(piece,U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval,piece))
    # Save the last trajectory:
    np.save(filename + "%d.npy"%q,ieval)
    file_names.append(filename + "%d.npy"%q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y.chunksize
    return reader

def DoubleProductsLinear(Y1,Y2,U,filename):
    ''' Evaluate all products between two given time-series and then compute a
    linear transformation of the product basis.
    
    Parameters:
    -------------
    Y1, Y2: pyemma-reader, containing time series of basis functions.
    U, ndarray, shape (r,s), where r must be identical to the product dimension
        of Y1 and Y2 and s is the number of linear combinations to be extracted.
    filename: str, name to be used to save the data for the product time series.
    
    Returns:
    -------------
    pyemma-reader, containing the time-series of all possible products between
        the basis functions in Y1 and Y2.'''
    # Get the dimensions of both time-series:
    r1 = Y1.dimension()
    r2 = Y2.dimension()
    # Compute the product dimension:
    r = r1*r2
    # Get the iterators for both time-series:
    I1 = Y1.iterator()
    I2 = Y2.iterator()
    # Prepare an empty array for the trajectory pieces:
    file_names = []
    q = 0
    ieval = np.zeros((0,r))
    # Compute the products chunk by chunk:
    for piece in zip(I1,I2):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = piece[0][1]
        piece1 = piece[1][1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "%d.npy"%q,ieval)
            file_names.append(filename + "%d.npy"%q)
            ieval = np.zeros((0,r))
            q += 1
        # Compute all the products:
        chunkeval = np.einsum('ijk,imk->ijm',piece0[:,:,np.newaxis],piece1[:,:,np.newaxis])
        chunkeval = np.reshape(chunkeval,(chunkeval.shape[0],r))
        # Apply linear transform:
        chunkeval = np.dot(chunkeval,U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval,chunkeval))
    # Save the last trajectory:
    np.save(filename + "%d.npy"%q,ieval)
    file_names.append(filename + "%d.npy"%q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y1.chunksize
    return reader
                    

def TPCorrelations(Y1,Y2,Y3,lagtimes):
    ''' This function evaluates all products between three given time-series and
    computes their correlation matrices at lagtimes given by lagtimes.
     
    Parameters:
    -------------
    Y1, Y2, Y3: pyemma-reader, containing time series of basis functions.
    lagtimes: ndarray, shape(nlag,), list of the lagtimes for the correlation
        matrices.
     
    Returns:
    -------------
    pyemma-reader, containing the time-series of all possible products between
        the basis functions in Y1, Y2 and Y3.'''
    # Get the number of lagtimes requested:
    nlag = lagtimes.shape[0]
    # Get the dimensions of all time-series:
    r1 = Y1.dimension()
    r2 = Y2.dimension()
    r3 = Y3.dimension()
    # Compute the product dimension:
    r = r1*r2*r3
    # Get the number of trajectories and trajectory lengths:
    ntraj = Y1.number_of_trajectories()
    tlens = Y1.trajectory_lengths()
    # Get the iterators for both time-series:
    I1 = Y1.iterator()
    I2 = Y2.iterator()
    I3 = Y3.iterator()
    # Prepare the correlation matrices:
    C = np.zeros((nlag,r,r))
    q = 0
    ieval = np.zeros((0,r))
    # Compute the products chunk by chunk:
    for piece in zip(I1,I2,I3):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = piece[0][1]
        piece1 = piece[1][1]
        piece2 = piece[2][1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            # Update the correlation matrices:
            for l in range(nlag):
                lag = lagtimes[l]
                C[l,:,:] += (1.0/(tlens[q]-lag))*np.dot(ieval[:-lag,:].transpose(),ieval[lag:,:])
            ieval = np.zeros((0,r))
            q += 1
        # Compute all the products and stack them underneath ieval:
        chunkeval = np.einsum('ijk,imk,ink->ijmn',piece0[:,:,np.newaxis],piece1[:,:,np.newaxis],piece2[:,:,np.newaxis])
        chunkeval = np.reshape(chunkeval,(chunkeval.shape[0],r))
        ieval = np.vstack((ieval,chunkeval))
    # Compute the averages of the corelation-matrices:
    C = C/ntraj
    return C
# 
# def RHEigenvalue(Eval,k,tau,sep,return_otrans=False):
#     ''' Solves Roothaan-Hall generalized eigenvalue problem for given time series.
#     
#     Parameters:
#     ------------
#     Eval: ndarray, SHAPE (T,nf), contains time series of nf basis functions over a
#         trajectory of length T.
#     k: int, the number of eigenvalues to be computed.
#     tau: int, the lag time to be used.
#     sep: ndarray, SHAPE (1,nreps), Array of separation point for the individual trajectory
#         pieces. nreps is the number of replicas.
#     return_otrans, bool (optional): Return transformation into orthonormal basis
#         w.r.t. weighted scalar product in addition to eigenvectors.
#         
#     Returns:
#     -----------
#     V: ndarray, SHAPE (nf,k), eigenvectors of RH-eigenvalue problem. If return_otrans
#         is given, V contains expansion coefficients w.r.t. orthonormal basis.
#     W: ndarray, SHAPE (nf,Rp), transformation into orthogonal basis. Only returned
#         if return_otrans is given.
#     D: ndarray, SHAPE (k,), eigenvalues of RH-eigenvalue-problem.
#     Ctau, C0: ndarray, SHAPE(nf,nf), the correlation matrices at times tau / 0.
#     '''
#     # Compute both correlation matrices:
#     Ctau = CorrelationMatrix(Eval,tau,sep)
#     C0 = CorrelationMatrix(Eval,0,sep)
#     # Symmetrize Ctau:
#     Ctau = 0.5*(Ctau + Ctau.transpose())
#     # Obtain orthogonal basis transformation:
#     Vp = OrthTrans(C0)
#     # Transform time-lagged correlation-matrix into this basis:
#     Ctau_trans = np.dot(Vp.transpose(),np.dot(Ctau,Vp))
#     # Solve standard eigenvalue problem:
#     D,V = scl.eigh(Ctau_trans)
#     # Get the k leading eigenvalues:
#     ind = np.argsort(np.real(D))
#     D = D[ind]
#     V = V[:,ind]
#     D = D[-k:]
#     V = V[:,-k:]
#     # Make returns:
#     if return_otrans:
#         return (V,Vp,D,Ctau,C0)
#     else:
#         V = np.dot(Vp,V)
#         return (V,D,Ctau,C0)
# 
# def OrthTrans(C0):
#     ''' Computes orthonormal basis transformation for a (possibly rank-deficient)
#     correlation matrix.
#     
#     Parameters:
#     ------------
#     C0: ndarray, SHAPE(N,N), input correlation matrix between basis functions.
#         
#     Returns:
#     -----------
#     Vp: ndarray, SHAPE (N,R), matrix of re-scaled eigenvectors of C0 corresponding
#         to non-zero eigenvalues. They are re-scaled s.t. Vp^T*C0*Vp = Id. The columns
#         of Vp span the space of the original basis functions.
#     '''
#     # Get the shape of C0:
#     N = C0.shape[0]
#     # Compute eigenvalue decomposition:
#     Lp,Vp = scl.eigh(C0)
#     # Perform numerical cut-off:
#     ep = np.finfo(np.double).eps
#     ind = Lp>(N*np.max(C0)*ep)
#     Lp = Lp[ind]
#     Vp = Vp[:,ind]
#     # Re-scale the eigenvectors:
#     Vp = np.dot(Vp,np.diag(np.sqrt(1/Lp)))
#     # Return the transformation:
#     return Vp
#     
#     
# def CorrelationMatrix(Eval,tau,sep):
#     ''' Compute correlation matrix of basis function time series..
#     
#     Parameters:
#     ------------
#     Eval: ndarray, SHAPE (T,nf), contains time series of nf basis functions over a
#         trajectory of length T.
#     tau: int, the lag time to be used.
#     sep: ndarray, SHAPE (1,nreps), Array of separation point for the individual trajectory
#         pieces. nreps is the number of replicas.
#     
#     Returns:
#     ------------
#     C: ndarray, SHAPE (nf,nf), the time-lagged correlation matrix.
#     '''
#     # Get the size basis:
#     nf = np.shape(Eval)[1]
#     # Transpose Eval:
#     Eval = Eval.transpose()
#     # Prepare the correlation matrix:
#     C = np.zeros((nf,nf))
#     # Get the number of replicas:
#     nreps = np.shape(sep)[0]-1
#     # Get their weights:
#     Tlen = 1.0*sep[-1]
#     w = (sep[1:] - sep[:-1])/Tlen
#     # Get the results:
#     for m in range(nreps):
#         mC = CorrelationMatrixPart(Eval[:,sep[m]:sep[m+1]],tau)
#         C = C + w[m]*mC
#     return C
#     
# def CorrelationMatrixPart(eval,tau):
#     ''' Evaluate the time-lagged correlation matrix for basis function time
#     series within parallel evaluation.
#     
#     Parameters:
#     -------------
#     eval: ndarray, SHAPE (nf,Tlen), each row is the evaluation of a basis function.
#     tau: integer, the lag time to be used.
#     
#     Returns:
#     -------------
#     C: ndarray, SHAPE (nf,nf), the time-lagged correlation matrix.
#     '''
#     # Get the trajectory length:
#     T = np.shape(eval)[1]
#     # Prepare the output:
#     C = (1.0/(T-tau))*np.dot(eval[:,0:T-tau],np.transpose(eval[:,tau:]))
#     return C
# 
# 
# def TimescaleTestPP(Eval,taus,K,sep,dt):
#     ''' Performs timescale test for given time-series of basis functions.
#     
#     Parameters:
#     -----------
#     Eval: ndarray, shape (T,nf), where T is the total number of time-steps and nf is
#         the number of basis functions.
#     taus: ndarray, shape (Ntau,), contains all lag times for the timescaletest. Note that
#         zero should not be included here, it will be computed automatically.
#     K: integer >= 2, the number of eigenvalues to be used for the timescaletest.
#     sep: ndarray, shape (), array of separation point for the individual trajectory
#             pieces.
#     dt: float, physical time-step of the simulation data that was used.
#     
#     Returns:
#     -----------
#     ts: ndarray, shape (K-1,Ntau), contains all implied timescales for K eigenvalues and Ntau
#         lagtimes.
#     '''
#     # get the number of lag times:
#     taus = np.hstack(([0],taus))
#     Ntau = np.shape(taus)[0]
#     # Prepare output:
#     ts = np.zeros((K-1,Ntau-1))
#     # Prepare list of correlation matrices:
#     Clist = []
#     # Loop over the lag times:
#     for j in range(Ntau):
#         # Get the next lagtime:
#         jtau = taus[j]
#         # Compute the correlation matrix:
#         C = CorrelationMatrix(Eval,jtau,sep)
#         # Add it to the list:
#         Clist.append(C)
#     # Get the first of the correlation matrices:
#     C0 = Clist.pop(0)
#     # Avoid linear dependence of basis functions:
#     V0 = OrthTrans(C0)
#     # Compute timescales for different lag times:
#     for m in range(Ntau-1):
#         # Get the next time-lagged correlation matrix:
#         Ctau = Clist.pop(0)
#         # Transform C, S becomes identity matrix:
#         Ctau = np.dot(V0.transpose(),np.dot(Ctau,V0))    
#         nfr = np.shape(Ctau)[0]
#         # Solve the eigenvalue problem:
#         Dm, _ = scl.eigh(Ctau,eigvals=(nfr-K,nfr-1))
#         # Save the timescales:
#         ts[:,m] = -dt*taus[m+1]/np.log(Dm[:-1])
#     return ts