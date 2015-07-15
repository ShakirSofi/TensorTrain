import numpy as np
import scipy.linalg as scl

def BestApprox(reader,I):
    ''' This function solves the best-approximation problem for the stationary
    distribution from the data.
    
    Parameters:
    ------------
    reader: pyemma-reader, containing the evaluation of all basis functions
        across the data.
    I: ndarray, shape(nf,nf), matrix of overlap integrals of the basis functions.
    '''
    # Get the number of basis functions and the number of trajectories from the
    # reader:
    nf = reader.dimension()
    ntraj = reader.number_of_trajectories()
    tl = reader.trajectory_lengths()
    It = reader.iterator()
    # Compute the mean values from the data:
    # This is the vector of the means:
    b = np.zeros(nf)
    # This array is for intermediate values:
    ib = np.zeros(nf)
    q = 0
    # Loop over the chunks:
    for piece in It:
        # Check if a new trajectory has been started:
        if piece[0] > q:
            ib /= tl[q]
            b += ib
            ib = np.zeros(nf)
            q += 1
        # Sum up the next piece:
        ib += np.sum(piece[1],axis=0)
    # Finish the last trajectory:
    ib /= tl[q]
    b += ib
    # Average b:
    b /= ntraj
    # After this, solve the linear system:
    v = scl.solve(I,b,sym_pos=True)
    # Return the result:
    return v
        