'''
Created on 19.01.2015

@author: marscher
'''
from .transformer import Transformer
from pyemma.util.linalg import eig_corr
from pyemma.util.annotators import doc_inherit

import numpy as np

__all__ = ['TICA']


class TICA(Transformer):

    r"""
    Time-lagged independent component analysis (TICA)

    Parameters
    ----------
    tau : int
        lag time
    output_dimension : int
        how many significant TICS to use to reduce dimension of input data
    epsilon : float
        eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of Eigenvalues define the size
        of the output.
    force_eigenvalues_le_one : boolean
        Compute covariance matrix and time-lagged covariance matrix such
        that the generalized eigenvalues are always guaranteed to be <= 1.

    Notes
    -----
    Given a sequence of multivariate data :math:`X_t`, computes the mean-free
    covariance and time-lagged covariance matrix:

    .. math::

        C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T (X_t + \tau - \mu)

    and solves the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i

    where :math:`r_i` are the independent components and :math:`\lambda_i` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math:: t_i = -\tau / \ln |\lambda_i|

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    """

    def __init__(self, lag, output_dimension, epsilon=1e-6, force_eigenvalues_le_one=False, mean=None):
        super(TICA, self).__init__()

        # store lag time to set it appropriatly in second pass of parametrize
        self._lag = lag
        self._output_dimension = output_dimension
        self._epsilon = epsilon
        self._force_eigenvalues_le_one = force_eigenvalues_le_one
        if mean is None:
            self._compute_mean = True
        else:
            self._compute_mean = False
        self.mu = mean

        # covariances
        self.cov = None
        self.cov_tau = None
        # mean
        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0
        self.eigenvalues = None
        self.eigenvectors = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        self._parametrized = False
        self._lag = new_tau

    @doc_inherit
    def describe(self):
        return "[TICA, tau = %i; output dimension = %i]" \
            % (self._lag, self._output_dimension)

    def dimension(self):
        """ output dimension"""
        return self._output_dimension

    @doc_inherit
    def _get_memory_per_frame(self):
        # temporaries
        dim = self.data_producer.dimension()

        mean_free_vectors = 2 * dim * self.chunksize
        dot_product = 2 * dim * self.chunksize

        return 8 * (mean_free_vectors + dot_product)

    @doc_inherit
    def _get_constant_memory(self):
        dim = self.data_producer.dimension()

        # memory for covariance matrices (lagged, non-lagged)
        cov_elements = 2 * dim ** 2
        mu_elements = dim

        # TODO: shall memory req of diagonalize method go here?

        return 8 * (cov_elements + mu_elements)

    @property
    def mean(self):
        return self.mu

    @doc_inherit
    def _param_init(self):
        dim = self.data_producer.dimension()
        assert dim > 0, "zero dimension from data producer"
        assert self._output_dimension <= dim, \
            ("requested more output dimensions (%i) than dimension"
             " of input data (%i)" % (self._output_dimension, dim))

        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0

        self.cov = np.zeros((dim, dim))
        self.cov_tau = np.zeros_like(self.cov)

        self._logger.info("Running TICA with tau=%i; Estimating two covariance matrices"
                          " with dimension (%i, %i)" % (self._lag, dim, dim))

        # create mean array and covariance matrices
        if self._compute_mean:
            self.mu = np.zeros(dim)
            return 0  # in zero'th pass don't request lagged data if we compute the mean
        else:
            return self._lag # if mean is not computed, start with lagged data

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        """
        Chunk-based parameterization of TICA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance and time-lagged covariance
        matrices are estimated. Finally, the generalized eigenvalue problem is solved to determine
        the independent components.

        :param X:
            coordinates. axis 0: time, axes 1-..: coordinates
        :param itraj:
            index of the current trajectory
        :param t:
            time index of first frame within trajectory
        :param first_chunk:
            boolean. True if this is the first chunk globally.
        :param last_chunk_in_traj:
            boolean. True if this is the last chunk within the trajectory.
        :param last_chunk:
            boolean. True if this is the last chunk globally.
        :param ipass:
            number of pass through data
        :param Y:
            time-lagged data (if available)
        :return:
        """
        if self._compute_mean and ipass == 0:
            # TODO: maybe use stable sum here, since small chunksizes
            # accumulate more errors
            self.mu += np.sum(X, axis=0, dtype=np.float64)
            self._N_mean += np.shape(X)[0]

            if last_chunk:
                self.mu /= self._N_mean
                self._logger.info("calculated mean.")

                # now we request real lagged data, since we are finished
                # with first pass
                return False, self._lag

        elif (self._compute_mean and ipass == 1) or (not self._compute_mean and ipass == 0):
            if self.trajectory_length(itraj, stride=stride) > self._lag:
                self._N_cov_tau += 2.0*np.shape(Y)[0]
                X_meanfree = X - self.mu
                Y_meanfree = Y - self.mu
                # update the time-lagged covariance matrix
                end = min(X_meanfree.shape[0], Y_meanfree.shape[0])
                self.cov_tau += 2.0*np.dot(X_meanfree[0:end].T, Y_meanfree[0:end])

                # update the instantaneous covariance matrix
                if self._force_eigenvalues_le_one:
                    # MSM-like counting
                    Zptau = self._lag-t  # zero plus tau
                    Nmtau = self.trajectory_length(itraj, stride=stride)-t-self._lag  # N minus tau

                    # restrict to valid block indices
                    size = X_meanfree.shape[0]
                    Zptau = min(max(Zptau, 0), size)
                    Nmtau = min(max(Nmtau, 0), size)

                    # update covariance matrix
                    start2 = min(Zptau, Nmtau)
                    end2 = max(Zptau, Nmtau)
                    self.cov += np.dot(X_meanfree[0:start2, :].T, X_meanfree[0:start2, :])
                    self._N_cov += start2

                    if Nmtau > Zptau:
                        self.cov += 2.0*np.dot(X_meanfree[start2:end2, :].T, X_meanfree[start2:end2, :])
                        self._N_cov += 2.0*(end2-start2)

                    self.cov += np.dot(X_meanfree[end2:, :].T, X_meanfree[end2:, :])
                    self._N_cov += (size-end2)
                else:
                    # traditional counting
                    self.cov += 2.0*np.dot(X_meanfree.T, X_meanfree)
                    self._N_cov += 2.0*np.shape(X)[0]

            else:
                self._logger.warning("trajectory nr %i too short, skipping it" % itraj)

            if last_chunk:
                self._logger.info("finished calculation of Cov and Cov_tau.")
                return True  # finished!

        return False  # not finished yet.

    @doc_inherit
    def _param_finish(self):
        if self._force_eigenvalues_le_one:
            assert self._N_cov == self._N_cov_tau, 'inconsistency in C(0) and C(tau)'

        # symmetrize covariance matrices
        self.cov = self.cov + self.cov.T
        self.cov *= 0.5

        self.cov_tau = self.cov_tau + self.cov_tau.T
        self.cov_tau *= 0.5

        # norm
        self.cov /= self._N_cov - 1
        self.cov_tau /= self._N_cov_tau - 1

        # diagonalize with low rank approximation
        self._logger.info("diagonalize Cov and Cov_tau")
        self.eigenvalues, self.eigenvectors = \
            eig_corr(self.cov, self.cov_tau, self._epsilon)
        self._logger.info("finished diagonalisation.")

    def _map_array(self, X):
        """Projects the data onto the dominant independent components.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        # TODO: consider writing an extension to avoid temporary Xmeanfree
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self._output_dimension])
        return Y
