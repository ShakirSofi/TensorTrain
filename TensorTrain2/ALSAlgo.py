import numpy as np

import TensorTrain2.Util as UT
import TensorTrain2.Optimize as OP

def RunALS(T,A):
    ''' Run the ALS scheme on the tensor T.
    
    Parameters:
    ------------
    T: BlockTTTensor.
    A: ALS-object, with all important quantities.
    
    Returns:
    ------------
    T: BlockTTTensor, the modified tensor after the iteration.
    A: ALS-object, the modified iteration object.
    
    '''
    print "Starting ALS."
    # Get dimension:
    d = T.d
    # Iterate:
    q = 0
    while 1:
        print "-------------"
        print "Iteration %d"%q
        print "-------------"
        # Perform one full sweep:
        T, A = ALSSweep(T, A)
        # Check for early termination:
        if A.J == []:
            print "Optimization failed."
            break
        # Check for convergence:
        Jfull = np.array(A.J)
        if q > 0:
           diffq = np.abs(Jfull[-2*(d-2):] - Jfull[-4*(d-2):-2*(d-2)])
           if np.max(diffq) < A.eps_iter:
               break
           else:
               print "Individual differences too large: %.5f"%np.max(diffq)
               print "Continue with next Iteration"
               print ""
        q += 1
    # Return results:
    return (T, A)


def ALSSweep(T,A):
    ''' Perform one up-and-down sweep through the tensor.
    
    Parameters:
    ------------
    T: BlockTTTensor.
    A: ALS-object, with all important quantities.
    
    Returns:
    ------------
    T: BlockTTTensor, the modified tensor after the iteration.
    A: ALS-object, the modified iteration object.
    '''
    # Initialise the new left interface by the basis reader for position 0:
    Yk = T.ComponentBasis(0)
    T.SetInterface(0,Yk)
    # Loop forward over the tensor:
    for k in range(T.d-2):
        print "Forward problem %d"%(k)
        # Load the next right interface:
        RIk = T.GetInterface(k+1)
        # Compute quadruple products:
        Qk = UT.DoubleProducts(Yk,RIk,A.filename)
        # Call the eigenvalue solver:
        print "Diagonalizing product basis:"
        eigv = UT.Diagonalize(Qk,A.tau,A.M)
        # Update reference timescales:
        print "Eigenvalues:"
        print eigv.d
        # Update timescales:
        A.UpdateTimescales(eigv.d)
        # Perform low-rank decomposition:
        print "Computing low-rank decomposition:"
        Up,L = OP.LowRank(eigv, Yk.dimension()/T.basissize[k], T.basissize[k], RIk.dimension(), A)
        # Stop the process if no low-rank decomposition was possible:
        if Up is None:
            A.J = []
            return (T,A) 
        # Update the rank:
        T.R[k] = Up.shape[1]
        # Update the objective function:
        A.UpdateObjective(L)
        print "Objective: %.9e"%A.Objective()
        # Update the least-squares errors:
        if k > 0:
            shapes = (T.R[k-1],T.basissize[k],T.basissize[k+1],T.R[k+1])
            res = UT.LeastSquaresTest(eigv.C0,shapes,Up)
            T.SetLSError(k,res)
        # Update component k:
        if k == 0:
            T.SetComponentTensor(k,np.reshape(Up,(1,T.basissize[k],T.R[k])))
        else:
            T.SetComponentTensor(k,np.reshape(Up,(T.R[k-1],T.basissize[k],T.R[k])))
        # Change the root:
        T.root = k+1
        # Apply the linear transform to Yk:
        Yk = UT.ApplyLinearTransform(Yk,Up,A.filename)
        # Take double products with the next basis set to generate the next
        # interface:
        Yk = UT.DoubleProducts(Yk,T.ComponentBasis(k+1),T.tensordir+"Interface%d"%(k+1))
        # Update interface:
        T.SetInterface(k+1,Yk)
        print ""
        print ""
    # Loop backward over the tensor:
    # Re-initialize Yk:
    Yk = T.ComponentBasis(T.d-1)
    T.SetInterface(T.d-1,Yk)
    for k in range(T.d-1,1,-1):
        print "Backward problem %d"%(k)
        # Load the next right interface:
        RIk = T.GetInterface(k-1)
        # Compute quadruple products:
        Qk = UT.DoubleProducts(Yk,RIk,A.filename)
        # Call the eigenvalue solver:
        print "Diagonalizing product basis:"
        eigv = UT.Diagonalize(Qk,A.tau,A.M)
        # Update reference timescales:
        print "Eigenvalues:"
        print eigv.d
        # Update timescales:
        A.UpdateTimescales(eigv.d)
        # Perform low-rank decomposition:
        print "Computing low-rank decomposition:"
        Up,L = OP.LowRank(eigv, Yk.dimension()/T.basissize[k], T.basissize[k], RIk.dimension(), A)
        # Stop the process if no low-rank decomposition was possible:
        if Up is None:
            A.J = []
            return (T,A)
        # Update the rank:
        T.R[k-1] = Up.shape[1]
        # Update the objective function:
        A.UpdateObjective(L)
        print "Objective: %.9e"%A.Objective()
        if k < T.d-1:
            shapes = (T.basissize[k],T.R[k],T.R[k-2],T.basissize[k-1])
            res = UT.LeastSquaresTest(eigv.C0,shapes,Up,backward=True)
            T.SetLSError(k,res)
        # Update component k:
        if k == T.d-1:
            T.SetComponentTensor(k,np.reshape(Up,(T.R[k-1],T.basissize[k],1)))
        else:
            T.SetComponentTensor(k,np.reshape(Up,(T.R[k-1],T.basissize[k],T.R[k])))
        # Change the root:
        T.root = k-1
        # Apply the linear transform to Yk:
        Yk = UT.ApplyLinearTransform(Yk,Up,A.filename)
        # Take double products with the next basis set to generate the next
        # interface:
        Yk = UT.DoubleProducts(T.ComponentBasis(k-1),Yk,T.tensordir+"Interface%d"%(k-1))
        # Update interface:
        T.SetInterface(k-1,Yk)
        print ""
        print ""
    return (T,A)