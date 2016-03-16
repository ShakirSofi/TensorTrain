import numpy as np
import numpy.linalg as npl
import scipy.special as scs
import matplotlib.pyplot as plt

import pyemma.coordinates as pco

import variational.solvers.direct as vsd
import variational.estimators.running_moments as vrm

''' Utility functions for TT-computations.'''


def ApplyLinearTransform(Y, U, filename):
    """ Apply linear transformation U to time-series given by Y.

    Parameters:
    -------------
    Y, pyemma-reader, containing time series of basis functions.
    U, ndarray, shape (r,s), where r must be identical to dimension of Y and s
        is the number of linear combinations to be extracted.
    filename: str, name to be used to save the data for the new time series.

    Returns:
    -------------
    pyemma-reader, containing the time-series of all the linear transform
        applied to Y."""
    # Get the dimension of the new time-series:
    r = U.shape[1]
    # Get the iterator for the time-series:
    I = Y.iterator()
    # Prepare an empty array for the trajectory pieces:
    file_names = []
    q = 0
    ieval = np.zeros((0, r))
    # Compute the products chunk by chunk:
    for piece in I:
        # Get the trajectory number and the data:
        traj_id = piece[0]
        piece = piece[1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "_%d.npy" % q, ieval)
            file_names.append(filename + "_%d.npy" % q)
            ieval = np.zeros((0, r))
            q += 1
        # Apply linear transform:
        piece = np.dot(piece, U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval, piece))
    # Save the last trajectory:
    np.save(filename + "_%d.npy" % q, ieval)
    file_names.append(filename + "_%d.npy" % q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y.chunksize
    return reader


def DoubleProducts(Y1, Y2, filename, U=None):
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
    r = r1 * r2
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
    ieval = np.zeros((0, ro))
    # Compute the products chunk by chunk:
    for piece in zip(I1, I2):
        # Get the trajectory number and the data:
        traj_id = piece[0][0]
        piece0 = piece[0][1]
        piece1 = piece[1][1]
        # Check if the last trajectory is finished:
        if traj_id > q:
            np.save(filename + "_%d.npy" % q, ieval)
            file_names.append(filename + "_%d.npy" % q)
            ieval = np.zeros((0, ro))
            q += 1
        # Compute all the products:
        chunkeval = np.einsum('ijk,imk->ijm', piece0[:, :, np.newaxis], piece1[:, :, np.newaxis])
        chunkeval = np.reshape(chunkeval, (chunkeval.shape[0], r))
        # Apply linear transform if necessary:
        if not (U is None):
            chunkeval = np.dot(chunkeval, U)
        # Stack the result underneath the previous results:
        ieval = np.vstack((ieval, chunkeval))
    # Save the last trajectory:
    np.save(filename + "_%d.npy" % q, ieval)
    file_names.append(filename + "_%d.npy" % q)
    # Build a new reader and return it:
    reader = pco.source(file_names)
    reader.chunksize = Y1.chunksize
    return reader


class DiagonalizationResult:
    """ This class is just a container for the results of a diagonalization problem.

    Parameters:
    -----------
    d: ndarray (M,)
        the eigenvalues computed.
    V: ndarray (N,M)
        the eigenvectors computed.
    Ct, C0: ndarray (N,N)
        the correlation matrices.
    """
    def __init__(self,d, V, Ct, C0):
        self.d = d
        self.V = V
        self.Ct = Ct
        self.C0 = C0


def Diagonalize(reader, tau, M):
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
    # Get output from the reader:
    out = reader.get_output()
    # Determine number of trajectories:
    ntraj = reader.number_of_trajectories()
    # Determine number of basis functions:
    nd = reader.dimension()
    # Instantiate estimator object:
    est = vrm.running_covar(xx=True, xy=True, symmetrize=True, sparse_mode='dense')
    # Add all trajectories:
    for n in range(ntraj):
        # Extract the next trajectory:
        ntraj = out[n]
        # Add to estimator:
        est.add(ntraj[:-tau, :], ntraj[tau:, :])
    # Get correlation matrices:
    Ct = est.cov_XY()
    C0 = est.cov_XX()
    # Plug them into estimator:
    d, V = vsd.eig_corr_qr(C0, Ct)
    # Select only the first M eigenvalue / eigenvector pairs:
    d = d[:M]
    V = V[:, :M]
    # Return results:
    eigv = DiagonalizationResult(d, V, Ct, C0)
    return eigv


''' Analysis functions:'''


def LeastSquaresTest(C, shapes, Up, backward=False):
    ''' Computes a least-squares approximation of the interface functions in
    terms of the previous interface, to measure the contribution of a single 
    coordinate.
    
    Parameters:
    -----------
    C: ndarray, instantaneous correlation matrix of full 4-fold product basis.
    shapes: tuple, with four entries, the dimensions of the bases in C0.
    Up: ndaray, the interfaces to be approximated.
    backward: bool, indicates that this is on the backward sweep, changes the
        order of indices in the arrays.
    '''
    # Reshape C0:
    C = np.reshape(C, shapes + shapes)
    # Extract the part of C0 that contains only the first two basis sets:
    C = C[:, :, 0, 0, :, :, 0, 0].copy()
    if backward:
        # Compute the least-squares matrix:
        A = C[0, :, 0, :].copy()
        # Compute the vector of right-hand sides:
        B = C[0, :, :, :].copy()
        B = np.reshape(B, (shapes[1], shapes[0] * shapes[1]))
    else:
        A = C[:, 0, :, 0].copy()
        B = C[:, 0, :, :].copy()
        B = np.reshape(B, (shapes[0], shapes[0] * shapes[1]))
    b = np.dot(B, Up)
    # Solve least-squares problems:
    c = npl.solve(A, b)
    # Compute the residuals:
    res = np.zeros(b.shape[1] - 1)
    for kp in range(1, b.shape[1]):
        res[kp - 1] = 1 - 2 * np.dot(c[:, kp], b[:, kp]) + np.dot(c[:, kp], np.dot(A, c[:, kp]))
    # Sum up the residuals and return them:
    if res.shape[0] > 0:
        return np.mean(res)
    else:
        return 0.0


def EvalEigenfunctions(T, tau, filename):
    ''' Evaluates the eigenfunctions of a TT-tensor and saves the results.
    
    Parameters:
    ------------
    T: BlockTTtensor-object.
    filename: string, the filename for the evaluation-files.
    
    Returns:
    pyemma-reader, containing all the evaluations.
    '''
    # Get the root:
    k = T.root
    # Get the right interface:
    Rk = T.GetInterface(k)
    # Get the left interface:
    Lk = T.GetInterface(k - 1)
    # Transform:
    Ukm = T.ComponentTensor(k - 1, order=1)
    # Apply it:
    Lk = ApplyLinearTransform(Lk, Ukm, filename)
    # Compute the double products with Rk:
    Rk = DoubleProducts(Lk, Rk, filename)
    # Diagonalize:
    eigv = Diagonalize(Rk, tau, T.M)
    # Get the eigenvectors:
    V = eigv.V
    # Multiply:
    Yk = ApplyLinearTransform(Rk, V, filename)
    return Yk


def CreateEVHistogram(ev_traj, bins, filename, m=np.array([1]), rg=None, kb=8.314e-3, T=300):
    ''' Create a histogram of the eigenfunction.
    
    Parameters:
    ------------
    ev_traj: List of eigenfunction trajectories.
    nbins: int, number of bins.
    m: Indices of eigenfunctions to be histogrammed: By default, the second ei-
    genfunction is shown. If m contains another integer, this function is shown.
    If m is a two-element array, a 2d-histogram of the two functions is shown.
    '''
    # Get the number of trajectories:
    ntraj = len(ev_traj)
    # Create a reader of eigenfunction data:
    ef = pco.source(ev_traj)
    ef.chunksize = np.min(ef.trajectory_lengths())
    # Create the histogram depending on m:
    if m.shape[0] == 1:
        psidata = ef.get_output(dimensions=m)
        psi = np.zeros((0, 1))
        # Stack all data on top of each other:
        for m in range(ntraj):
            psi = np.vstack((psi, psidata[m]))
        # Show the histogram:
        plt.figure()
        plt.hist(psi, bins=bins, range=rg)
    elif m.shape[0] == 2:
        psidata = ef.get_output(dimensions=m)
        psi = np.zeros((0, 2))
        # Stack all data on top of each other:
        for m in range(ntraj):
            psi = np.vstack((psi, psidata[m]))
        # Show the histogram: 
        plt.figure()
        H, xe, ye = np.histogram2d(psi[:, 0], psi[:, 1], bins=bins, range=rg, normed=True)
        # Make it a free energy plot:
        binwx = xe[1] - xe[0]
        binwy = ye[1] - ye[0]
        H = H * binwx * binwy
        ind = np.nonzero(H)
        thres = np.min(H[ind[0], ind[1]])
        H2 = thres * np.ones(H.shape)
        H2[ind[0], ind[1]] = H[ind[0], ind[1]]
        H2 = -kb * T * np.log(H2)
        X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]))
        plt.contourf(X, Y, H2.transpose())
        plt.colorbar()
    else:
        print "Selection in m could not be used."
    plt.savefig(filename)
    plt.show()


def SaveEVFrames(dt, ev_traj, c, d, traj_inp=None, filename=None, topfile=None, nframes=None):
    ''' Save frames that correspond to eigenvector centers from md-trajectories
    to separate trajectory.
    
    Parameters:
    --------------
    traj_inp: List of underlying md-trajectories.
    ev_traj: List of eigenfunction trajectories.
    dt: Physical time step.
    c: ndarray, shape(nc,M), centers.
    d: ndarray, shape(nc,). admissible distances to the centers.
    filename: str, name of the center-trajectories
    topfile:str, topology-file
    nframes: int, number of frames per center and per trajectory.
    '''
    # Get the number of trajectories:
    ntraj = len(ev_traj)
    # Get the number of centers and eigenfunctions:
    nc, M = c.shape
    # Create a reader of eigenfunction data:
    ef = pco.source(ev_traj)
    ef.chunksize = np.min(ef.trajectory_lengths())
    # Get the output into memory, leaving out the first ef:
    psidata = ef.get_output(dimensions=np.arange(1, M + 1, dtype=int))
    cindices = []
    # Write out frames to a trajectory file:
    # Loop over the centers:
    for i in range(nc):
        # Create a list of possible frames:
        indices = []
        # Loop over the trajectory files:
        for m in range(ntraj):
            # Get the data for this traj:
            mdata = psidata[m]
            # Get the admissible frames for this trajectory:
            mind = np.where(np.any(np.abs(mdata - c[i, :]) <= d[i], axis=1))[0]
            # Make a random selection:
            if not (nframes is None):
                mind = dt * np.random.choice(mind, (nframes,))
            else:
                mind = dt * mind
            # Put the information together:
            mindices = np.zeros((mind.shape[0], 2), dtype=int)
            mindices[:, 0] = m
            mindices[:, 1] = mind
            indices.append(mindices)
        # Save to traj:
        if not (traj_inp is None) and not (filename is None) and not (topfile is None):
            pco.save_traj(traj_inp, indices, outfile=filename + "Center%d.xtc" % i, topfile=topfile)
        cindices.append(indices)
    return cindices


''' Basis Set Definitions.'''


def EvalGaussian(x, mu, sig):
    ''' Evaluates Gaussian basis function over data x.
    
    Parameters:
    ------------
    x: ndarray, shape(T,), the data.
    mu, sig: ndarray, shape(N,), the mean values and standard deviations for
    the Gaussians.
    
    Returns:
    ndarray, shape(T,N), evaluation of all Gaussians.
    '''
    # Get the number of Gaussians:
    N = mu.shape[0]
    # Get the size of the data:
    T = x.shape[0]
    # Prepare output:
    y = np.zeros((T, N))
    # Evaluate the functions one by one:
    for n in range(N):
        y[:, n] = (1.0 / (np.sqrt(2 * np.pi) * sig[n])) * np.exp(-(x - mu[n]) ** 2 / (2 * sig[n] ** 2))
    return y


def EvalGaussianAngle(phi, mu, sig, normalized=False):
    ''' Evaluates a Gaussian basis function on 2pi-peridic domain. The input
    angles are transformed to their sin/cos-coordinates first.
     
    Parameters:
    ------------
    phi, ndarray, shape(T,), array of angles.
    mu, sig: ndarray, shape(N,), the mean values and standard deviations for
    the Gaussians.
    normalized: bool, return the normalized Gaussian probability function.
    
    Returns:
    ndarray, shape(T,N), evaluation of all Gaussians.
    '''
    # Get the number of Gaussians:
    N = mu.shape[0]
    # Get the size of the data:
    T = phi.shape[0]
    # Prepare output:
    y = np.zeros((T, N))
    # Transform the array:
    x = np.zeros((T, 2))
    x[:, 0] = np.cos(phi)
    x[:, 1] = np.sin(phi)
    # Also transform mu and sig:
    mu = np.array([np.cos(mu), np.sin(mu)])
    mu = mu.transpose()
    # Evaluate the Gaussians one by one_:
    for n in range(N):
        xn = x - mu[n, :]
        nsig = (1.0 / sig[n]) * np.eye(2)
        nyvec = np.einsum('ij,ji->i', xn, np.dot(nsig, xn.transpose()))
        if normalized:
            y[:, n] = (1.0 / (2 * np.pi * sig[n])) * np.exp(-0.5 * nyvec)
        else:
            y[:, n] = np.exp(-0.5 * nyvec)
    return y


def EvalFourier(x, M, normalized=False):
    ''' Evaluates all real Fourier basis functions up to order M over 1-d-array
    x.
    
    Parameters:
    ------------
    x: ndarray, shape(T,), data points.
    M: int, highest frequency, i.e. all functions including sin(Mx) and cos(Mx)
        are evaluated.
    normalized: bool, evaluate Fourier waves normalized as an orthonormal basis.
        
    Returns:
    ------------
    ndarray, shape(T,2*M+1)
    '''
    # Determine pre-factors:
    if normalized:
        n0 = 1.0 / np.sqrt(2 * np.pi)
        n1 = 1.0 / np.sqrt(np.pi)
    else:
        n0 = 1
        n1 = 1
    # Get the data size:
    T = x.shape[0]
    # Prepare output:
    y = np.zeros((T, 2 * M + 1))
    # Evaluate:
    y[:, 0] = n0
    for m in range(1, M + 1):
        y[:, 2 * m - 1] = n1 * np.sin(m * x)
        y[:, 2 * m] = n1 * np.cos(m * x)
    return y


def EvalLegendre(x, M):
    ''' Evaluates all Legendre polynomials up to order M over 1-d-array x.
    
    Parameters:
    ------------
    x: ndarray, shape(T,), data points.
    M: int, highest order of Legendre polynomials to be used
        
    Returns:
    ------------
    ndarray, shape(T,M+1)
    '''
    # Create output:
    T = x.shape[0]
    y = np.zeros((T, M + 1))
    # Evaluate the polynomials sequentially:
    for m in range(M + 1):
        # Get the coefficients:
        p = scs.legendre(m)
        # Evaluate:
        y[:, m] = np.polyval(p, x)
    return y
