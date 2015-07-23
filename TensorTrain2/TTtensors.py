import numpy as np

import Util as UT

class BlockTTtensor:
    def __init__(self,U,basis,M,tensordir="./"):
        ''' This class serves for the representation of several high-dimensional functions
        in the block-TT-format.
        
        Parameters:
        ------------
        U: List of component tensors, LENGTH d, where d is the dimension of the tensor. The first element
            should be a fourth order tensor. Its last index yields the number of functions represented by
            the block-TT-tensor. All others are third order tensors.
        basis: A list of pyemma-reader objects. Each reader contains the evaluation of the one-coordinate basis
            for one specific coordinate.
        M: Number of eigenfunctions encoded by the block tensor.
        tensordir: String, directory where intermediate evaluations are stored. Defaults to the current directory.
        '''
        # Make the inputs generally known:
        self.U = U
        self.basis = basis
        self.tensordir = tensordir
        self.M = M
        # Get the dimensions of the tensor:
        self.d = len(self.U)
        # Define the root:
        self.root = 0
        # Extract the ranks:
        self.R = np.zeros(self.d-1,dtype=int)
        for k in range(self.d-1):
            self.R[k] = self.U[k].shape[2]
        # Define a list of basis sizes:
        self.basissize = [self.basis[i].dimension() for i in range(self.d)]
        # Define a list of least-squares errors (for analysis):
        self.lse = np.zeros(self.d-2)
        # Compute the initial interfaces:
        self.interfaces = self.InitInterfaces()
    def GetRanks(self,k=-1):
        ''' Return k-th TT-rank or a vector of all TT-ranks, if k is not given.
        
        Parameters:
        ------------
        k: int, the index of the rank which is required. If k is not given, a
            vector of all ranks is returned.
        
        Returns
        ----------
        R: int or nd-array(self.d-1,).
        '''
        if k>=0 or k<=(self.d-2):
            return self.R[k]
        else:
            return self.R
    def ComponentBasis(self,k):
        ''' Return the component time series for component k
        
        Parameters:
        -------------
        k, integer, the component for which the time series is required.
        
        Returns:
        -------------
        pyemma-reader, the time series for this component. 
        '''
        return self.basis[k]
    def ComponentTensor(self,k,order=0):
        ''' Return the component tensor for component k
        
        Parameters:
        -------------
        k, integer, the component whose component tensor is required.
        order: Can be 0,1,2. order=0 means that the component is returned as a
            third order tensor, while order=1 returns the left unfolding and
            order=2 returns the right unfolding. However, this will be ignored
            if k is the root.
        
        Returns:
        -------------
        U: ndarray, SHAPE (r_{p-1},n_p,r_p), the component tensor for
            component k.
        '''
        if k == 0:
            rkm = 1
            rkp = self.R[k]
        elif k == self.d-1:
            rkm = self.R[k-1]
            rkp = 1
        else:
            rkm = self.R[k-1]
            rkp = self.R[k]
        if order == 1 and not k == self.root:
            return np.reshape(self.U[k],(rkm*self.basissize[k],rkp))
        elif order == 2 and not k == self.root:
            return np.reshape(self.U[k],(rkm,self.basissize[k]*rkp))
        else:
            return self.U[k]
    def SetComponentTensor(self,k,U):
        ''' Set the component tensor for component k
        
        Parameters:
        -------------
        k, integer, the component whose component tensor is required.
        U: ndarray, SHAPE (r_{p-1},n,r_p), the new component tensor for
            component k.
        '''
        self.U[k] = U
    def SetLSError(self,k,e):
        ''' Update the least-squares error for a given component.
        
        Parameters:
        -----------
        k: int, the entry to be updated.
        e: float, the new least-squares error.
        '''
        self.lse[k-1] = e
    def GetInterface(self,k):
        ''' Get the interface for component k.
        
        Parameters;
        ------------
        k, integer, the component whose interface is required.
        
        Returns:
        ------------
        pyemma-reader, the interface at component k.
        '''
        return self.interfaces[k]
    def SetInterface(self,k,reader):
        ''' Set the interface for component k.
        
        Parameters;
        ------------
        k, integer, the component whose interface is to be renewed.
        reader: pyemma-reader, the new interface.
        '''
        self.interfaces[k] = reader
    def InitInterfaces(self):
        ''' This function evaluates all the interfaces after initialisation of
        the block tensor. 
             
        Returns:
        ------------
        List of pyemma-readers, containing the interfaces:
        '''
        interface_list = []
        # Initialize by the basis reader of position d:
        Yk = self.ComponentBasis(self.d-1)
        interface_list.insert(0,Yk)
        # Loop from prelast position down to position zero.
        for k in range(self.d-2,-1,-1):
            # Get the component tensor for this coordinate:
            Ukp = self.ComponentTensor(k+1,order=2)
            # Apply component tensor:
            Yk = UT.ApplyLinearTransform(Yk,Ukp.transpose(),self.tensordir+"Interface%d"%k)
            # Compute double products with next basis:
            Fk = self.ComponentBasis(k)
            Yk = UT.DoubleProducts(Fk,Yk,self.tensordir+"Interface%d"%k)
            interface_list.insert(0,Yk)
        return interface_list