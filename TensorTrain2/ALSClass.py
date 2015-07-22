import numpy as np

class ALS:
    def __init__(self,tau,dt,M,filename,rmax,tol,gtol=1e-3,eps_iter=1e-2):
        ''' Creates an ALS-object that keeps track of several important quanti-
        ies during an ALS-iteration.
        
        Parameters:
        -------------
        tau: int, the lag-time to perform eigenvalue optimization.
        dt: int, physical time step.
        M: int, the number of eigenvalue / -function pairs.
        filename: str, the name for intermediate basis evaluation files.
        rmax: int, maximal rank allowed during the iteration. ALS process will
            terminate without results if this rank is exceeded.
        tol: float: tolerance for acceptance of low-rank step.
        gtol: float, stopping tolerance for CG-minimization in rank adaption step.
        eps_iter, float, convergence tolerance for overall iteration.
        '''
        # Make important quantities know:
        self.tau = tau
        self.dt = dt
        self.eps_iter = eps_iter
        self.rmax = rmax
        self.M = M
        self.filename = filename
        self.tol = tol
        self.gtol = gtol
        # Initialize objective value and timescales:
        self.J = []
        self.ts = np.zeros((self.M-1,0))
        self.Lref = 0
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
    def UpdateTimescales(self,ev):
        ''' Updates Timescale array.
        
        Parameters:
        ------------
        ev: ndarray, shape(M,), array of eigenvalues.
        '''
        # Compute timescales:
        ts = -self.dt*self.tau/np.log(ev[1:self.M])
        self.ts = np.hstack((self.ts,ts[:,None]))
    def Timescales(self):
        ''' Returns timescale array.'''
        return self.ts
    def UpdateLref(self,lref):
        ''' Updates the reference value for the objective function.'''
        if lref < self.Lref:
            self.Lref = lref