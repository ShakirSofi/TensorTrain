import numpy as np
import pyemma.coordinates as pco

def Reweight(f,pi,filename,minval):
    ''' This functions reweights a given basis trajectory by the inverse square
    root of the stationary distribution pi.
    
    Parameters:
    ------------
    f: pyemma-reader, contains the evaluation of the basis functions.
    pi: pyemma-reader, contains the evaluation of the stationary distribution.
    filename: str, filename for the evaluation files to be produced.
    minval: float, minimal value (greater than zero) allowed for the stationary
        distribution. All values smaller than minval are replaced by minval.
    
    Returns:
    -----------
    pyemma-reader, the reweighted basis.
    '''
    # Get the basis set size:
    r0 = f.dimension()
    # Get the iterators for both time-series:
    I1 = f.iterator()
    I2 = pi.iterator()
    # Prepare an empty array for the trajectory pieces:
    file_names = []
    q = 0
    ieval = np.zeros((0,r0))
    # Compute the products chunk by chunk:
    for piece in zip(I1,I2):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = np.copy(piece[0][1])
        piece1 = np.copy(piece[1][1])
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "_%d.npy"%q,ieval)
            file_names.append(filename + "_%d.npy"%q)
            ieval = np.zeros((0,r0))
            q += 1
        # Reweight:
        # Replace too small and negative values:
        minind = piece1[:,0] < minval
        piece1[minind,:] = minval
        # Re-weight the basis functions:
        piece0 = piece0/np.sqrt(piece1)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval,piece0))
    # Save the last trajectory:
    np.save(filename + "_%d.npy"%q,ieval)
    file_names.append(filename + "_%d.npy"%q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = f.chunksize
    return reader 