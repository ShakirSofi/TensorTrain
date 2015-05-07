import numpy as np

import pyemma.coordinates as pco
from pyemma.coordinates.transform.tica import TICA

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
            np.save(filename + "_%d.npy"%q,ieval)
            file_names.append(filename + "_%d.npy"%q)
            ieval = np.zeros((0,r))
            q += 1
        # Apply linear transform:
        piece = np.dot(piece,U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval,piece))
    # Save the last trajectory:
    np.save(filename + "_%d.npy"%q,ieval)
    file_names.append(filename + "_%d.npy"%q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y.chunksize
    return reader

def DoubleProducts(Y1,Y2,filename,U=None):
    ''' Evaluate all products between two given time-series. Optionally,a
    linear transformation of the product basis can be computed instead.
    
    Parameters:
    -------------
    Y1, Y2: pyemma-reader, containing time series of basis functions.
    filename: str, name to be used to save the data for the product time series.
    U, ndarray, shape (r,s), where r must be identical to the product dimension
        of Y1 and Y2 and s is the number of linear combinations to be extracted.
    
    Returns:
    -------------
    pyemma-reader, containing the time-series of all possible products between
        the basis functions in Y1 and Y2.'''
    # Get the dimensions of both time-series:
    r1 = Y1.dimension()
    r2 = Y2.dimension()
    # Compute the product dimension:
    r = r1*r2
    # Get the output dimension:
    if not (U is None):
        ro = U.shape[1]
    else:
        ro = r
    # Get the iterators for both time-series:
    I1 = Y1.iterator()
    I2 = Y2.iterator()
    # Prepare an empty array for the trajectory pieces:
    file_names = []
    q = 0
    ieval = np.zeros((0,ro))
    # Compute the products chunk by chunk:
    for piece in zip(I1,I2):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = piece[0][1]
        piece1 = piece[1][1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "_%d.npy"%q,ieval)
            file_names.append(filename + "_%d.npy"%q)
            ieval = np.zeros((0,ro))
            q += 1
        # Compute all the products:
        chunkeval = np.einsum('ijk,imk->ijm',piece0[:,:,np.newaxis],piece1[:,:,np.newaxis])
        chunkeval = np.reshape(chunkeval,(chunkeval.shape[0],r))
        # Apply linear transform if necessary:
        if not (U is None):
            chunkeval = np.dot(chunkeval,U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval,chunkeval))
    # Save the last trajectory:
    np.save(filename + "_%d.npy"%q,ieval)
    file_names.append(filename + "_%d.npy"%q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y1.chunksize
    return reader
                    

def TripleProducts(Y1,Y2,Y3,filename):
    ''' This function evaluates all products between three given time-series.
    Note that the evaluations of the triple-products have to be stored in disc
    at the moment.
     
    Parameters:
    -------------
    Y1, Y2, Y3: pyemma-reader, containing time series of basis functions.
    filename: str, name for intermediate files.
     
    Returns:
    -------------
    pyemma-reader, containing the evaluations of all triple-product basis
    functions.
    '''
    # Get the dimensions of all time-series:
    r1 = Y1.dimension()
    r2 = Y2.dimension()
    r3 = Y3.dimension()
    # Compute the product dimension:
    r = r1*r2*r3
    # Get the iterators for both time-series:
    I1 = Y1.iterator()
    I2 = Y2.iterator()
    I3 = Y3.iterator()
    # Prepare an array for the evaluations:
    q = 0
    ieval = np.zeros((0,r))
    file_names = []
    # Compute the products chunk by chunk:
    for piece in zip(I1,I2,I3):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = piece[0][1]
        piece1 = piece[1][1]
        piece2 = piece[2][1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "_%d.npy"%q,ieval)
            file_names.append(filename + "_%d.npy"%q)
            ieval = np.zeros((0,r))
            q += 1
        # Compute all the products and stack them underneath ieval:
        chunkeval = np.einsum('ijk,imk,ink->ijmn',piece0[:,:,np.newaxis],piece1[:,:,np.newaxis],piece2[:,:,np.newaxis])
        chunkeval = np.reshape(chunkeval,(chunkeval.shape[0],r))
        ieval = np.vstack((ieval,chunkeval))
    # Save the last trajectory:
    np.save(filename + "_%d.npy"%q,ieval)
    file_names.append(filename + "_%d.npy"%q)
    # Construct the reader for the triple-product basis:
    reader = pco.source(file_names)
    reader.chunksize = Y1.chunksize
    return reader
 
def Diagonalize(reader,tau,M):
    ''' Diagonalizes the generalized eigenvalue problem for the time-series re-
    presented by reader.
      
    Parameters:
    ------------
    reader: pyemma-reader, representing the time-series of some basis.
    tau: int, the lagtime to be used.
    M: int, the number of dominant eigenvalue / -vector pairs to be computed.
          
    Returns:
    -----------
    pyemma-TICA-object, from which all important information can be extracted.
    '''
    # Instantiate TICA-object. It is important to set mean=0 here, otherwise the
    # mean would be substracted from the time-series.
    tica = TICA(tau,M,epsilon=1e-14,mean=0)
    # Construct the stages of the pipeline:
    stages = [reader,tica]
    # Run the pipeline:
    pco.pipeline(stages)
    # Restrict the eigenvalue and eigenvectors to the requested number.
    tica.eigenvalues = tica.eigenvalues[:M]
    tica.eigenvectors = tica.eigenvectors[:,:M]
    return tica

def EvalFourier(x,M):
    ''' Evaluates all real Fourier basis functions up to order M over 1-d-array
    x.
    
    Parameters:
    ------------
    x: ndarray, shape(T,), data points.
    M: int, highest frequency, i.e. all functions including sin(Mx) and cos(Mx)
        are evaluated.
        
    Returns:
    ------------
    ndarray, shape(T,2*M+1)
    '''
    # Get the data size:
    T = x.shape[0]
    # Prepare output:
    y = np.zeros((T,2*M+1))
    # Evaluate:
    y[:,0] = 1
    for m in range(1,M+1):
        y[:,2*m-1] = np.sin(m*x)
        y[:,2*m] = np.cos(m*x)
    return y