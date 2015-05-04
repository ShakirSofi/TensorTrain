import numpy as np

class CanonicalTensor:
    ''' Structure for a 2-dimensional tensor in the canonical format. Pro-
        vides methods to set and access its components, get a full representa-
        tion, and compute derivatives of the associated quadratic form in the 
        matrix case.
        
        Parameters:
        R: int, tensor-rank.
        n0,n1: float, the subspace dimensions.
    '''
    def __init__(self,R,n0,n1):
        # Set all the global parameters:
        self.R = R
        self.n0 = n0
        self.n1 = n1
        self.d = 2
        # Define the main objects:
        self.U0 = np.zeros((self.R,self.n0))
        self.U1 = np.zeros((self.R,self.n1))
        self.fulldim = self.R*(self.n0+self.n1)
    def GetComponent(self,nu,r):
        ''' Return specific component of the tensor.
        
        Parameters:
        nu: int, dimension of the component within elem. tensor r.
        r: int, number of the elementary tensor for this component.
        
        Returns:
        U_rn: nd-array, shape(n_nu,), n_nu is the dimension of the nu-th
        subspace.
        '''
        if nu == 0:
            return self.U0[r,:]
        else:
            return self.U1[r,:]
    def GetDim(self,nu):
        ''' Return all coefficients for one dimension as a single vector:
         
         Parameters:
         nu: int, the dimension to be extracted.
         
         Returns:
         U_nu: nd-array, shape(R,n_nu), where n_nu is the subspace dimension.
        '''
        if nu == 0:
            return self.U0
        else:
            return self.U1
    def GetAll(self):
        ''' Return the full tensor as a single vector:
         
         Returns:
         U_full: nd-array, shape(R*sum(n_nu)), where n_nu are the subspace-
         dimensions.
        '''
        U_full = np.zeros(self.fulldim)
        U_full[:self.R*self.n0] = np.reshape(self.U0,(self.R*self.n0,))
        U_full[self.R*self.n0:] = np.reshape(self.U1,(self.R*self.n1,))
        return U_full
    def GetVU(self):
        ''' Compute full high-dimensional representation of the tensor:
        
        Returns:
        VU: nd-array, shape(n1*n2,)
        '''
        VU = np.dot(self.U0.transpose(),self.U1)
        VU = np.reshape(VU,(self.n0*self.n1,))
        return VU
    def SetComponent(self,nu,r,u):
        ''' Set one component to a new value:
        
        Parameters:
        nu: int, dimension of the component within elem. tensor r.
        r: int, number of the elementary tensor for this component.
        u: nd-array, shape(n_nu,), the new component.
        '''
        if nu ==0:
            self.U0[r,:] = u
        else:
            self.U1[r,:] = u
    def SetDim(self,nu,U):
        ''' Set one dimension component to a new value:
        
        Parameters:
        nu: int, dimension.
        U: nd-array, shape(R,n_nu), the new component.
        '''
        if nu ==0:
            self.U0[:,:] = U
        else:
            self.U1[:,:] = U
    def SetAll(self,U0,U1):
        ''' Set the entire tensor to a new value:
        
        Parameters;
        U0,U1: nd-array, shape(R,n0) and shape(R,n1), the new components.
        '''
        self.U0[:,:] = U0
        self.U1[:,:] = U1
    def PartialDerivative(self):
        ''' Return the partial derivatives of the map that takes the low-dimen-
        sional components to the high-dimensional full tensor.
        
        Returns:
        DU: nd-array, shape(n0*n1,R*(n0+n1)), matrix of derivative vectors for
        all low-dimensional components.
        '''
        # Prepare output:
        DU = np.zeros((self.n0*self.n1,self.R*(self.n0+self.n1)))
        # First dimension:
        for r in range(self.R):
            E = np.eye(self.n0,self.n0)
            vr = self.U1[r,:,None]
            vr = vr.transpose()
            for k in range(self.n0):
                ek = E[k,:,None]
                ekr = np.dot(ek,vr)
                DU[:,r*self.n0+k] = np.reshape(ekr,(self.n0*self.n1,))
        # Second dimension:
        offset = self.R*self.n0
        for r in range(self.R):
            E = np.eye(self.n1,self.n1)
            vr = self.U0[r,:,None]
            for k in range(self.n1):
                ek = E[k,:,None]
                ek = ek.transpose()
                ekr = np.dot(vr,ek)
                DU[:,offset+r*self.n1+k] = np.reshape(ekr,(self.n0*self.n1,))
        return DU