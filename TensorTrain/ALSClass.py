import numpy as np

class ALS:
    def __init__(self,tau,M,filename,eps_iter=1e-2,eps_orth=1e-8,eps_rank=0.9,mu0=10):
        ''' Creates an ALS-object that keeps track of several important quanti-
        ies during an ALS-iteration.
        
        Parameters:
        -------------
        tau: int, the lag-time to perform eigenvalue optimization.
        M: int, the number of eigenvalue / -function pairs.
        filename: str, the name for intermediate basis evaluation files.
        eps_iter, float, convergence tolerance for overall iteration.
        eps_orth: float, tolerance for orthogonality of solutions.
        eps_rank: float, tolerance for acceptance of implied timescales.
        mu0: float, initial penalty parameter for low-rank optimization.
        '''
        # Make important quantities know:
        self.tau = tau
        self.eps_iter = eps_iter
        self.eps_orth = eps_orth
        self.eps_rank = eps_rank
        self.mu0 = mu0
        self.M = M
        self.filename = filename
        # Initialize objective value:
        self.J = []
        # Initialize reference timescales:
        self.ts = np.zeros(self.M-1)
    def UpdateTS(self,ts,ev=False):
        ''' Set reference implied timescales to a new value.
        
        Parameters:
        -------------
        ts: ndarray, shape(self.M,), the new timescales (or eigenvalues).
        ev: If True, then ts is understood as an array of eigenvalues and con-
            verted before updating.
        '''
        # Compute implied timescales if needed:
        if ev:
            ts = -self.tau/np.log(ts[1:])
        # Update if necessary:
        self.ts = np.maximum(self.ts,ts)
    def RefTS(self):
        ''' Returns
        ------------
        self.ts, ndarray, shape(self.M,), the array of reference timescales.
        '''
        return self.ts
    def UpdateObjective(self,Jnew):
        ''' Update objective function value.
        
        Parameters:
        ------------
        Jnew: float, the new objective function.
        '''
        self.J.append(Jnew)
    def Objective(self):
        ''' Returns
        -------------
        float, current objective value.
        '''
        return self.J[-1]