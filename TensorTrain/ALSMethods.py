import numpy as np

import TensorTrain.Util as UT
import TensorTrain.LowRankMethods as LRM

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
    # Initialize objective function:
    J = 0
    q = 0
    # Iterate:
    while q < 1:
        print "-------------"
        print "Iteration %d"%q
        print "-------------"
        # Perform one full sweep:
        T,A = ALSSweep(T,A)
        # Check for convergence:
        Jq = A.Objective()
        if np.abs(Jq - J) < A.eps_iter:
            break
        else:
            J = Jq
            q += 1
    # Return the eigenvalues:
    return (T,A)


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
    # Loop forward over the tensor:
    for k in range(T.d-1):
        print "Forward problem %d"%(k)
        # Get the readers that are needed:
        Y2 = T.ComponentBasis(k)
        Y3 = T.GetInterface(k)
        if k == 0:
            Yk = UT.DoubleProducts(Y2,Y3,A.filename)
        else:
            Y1 = T.GetInterface(k-1)
            Yk = UT.TripleProducts(Y1,Y2,Y3,A.filename)
        # Call the eigenvalue solver:
        eigv = UT.Diagonalize(Yk,A.tau,A.M)
        # Update reference timescales:
        print "Eigenvalues:"
        print eigv.eigenvalues
        A.UpdateTS(eigv.eigenvalues,ev=True)
        # Perform low-rank decomposition:
        dims = T.GetRankTriple(k)
        LR = LRM.LowRank(eigv,dims,A)
        # Update reference timescales again:
        A.UpdateTS(LR.Timescales(A.tau))
        # Get the results:
        Ul = LR.GetComponents()
        # Update interface:
        if k == 0:
            Ik = UT.ApplyLinearTransform(Y2,Ul[0],T.tensordir+"Interface%d"%k)
        else:
            Ik = UT.DoubleProducts(Y1,Y2,T.tensordir+"Interface%d"%k,Ul[0])
        T.SetInterface(k,Ik)
        # Update component k:
        T.SetComponentTensor(k,np.reshape(Ul[0],(dims[0],dims[1],LR.R)))
        # Update next component:
        Ukplus = T.ComponentTensor(k+1)
        Uk2 = np.reshape(Ul[1].transpose(),(LR.R,dims[2],LR.M))
        Ukplus = np.einsum('ijk,jlm->ilmk',Uk2,Ukplus)
        T.SetComponentTensor(k+1,Ukplus)
        # Change the root:
        T.root = k+1
        # Store the objective function value:
        A.UpdateObjective(LR.Objective())
    # Loop backward over the tensor:
    for k in range(T.d-1,0,-1):
        print "Backward problem %d"%(k)
        # Get the readers that are needed:
        Y1 = T.GetInterface(k-1)
        Y2 = T.ComponentBasis(k)
        if k == T.d-1:
            Yk = UT.DoubleProducts(Y1,Y2,A.filename)
        else:
            Y3 = T.GetInterface(k)
            Yk = UT.TripleProducts(Y1,Y2,Y3,A.filename)
        # Call the eigenvalue solver:
        eigv = UT.Diagonalize(Yk,A.tau,A.M)
        # Update reference timescales:
        A.UpdateTS(eigv.eigenvalues,ev=True)
        print "Eigenvalues:"
        print eigv.eigenvalues
        # Perform low-rank decomposition. On the backward iteration, some indi-
        # ces must be swapped before the decomposition can be performed:
        dims = T.GetRankTriple(k)
        Up = eigv.eigenvectors
        Up = np.reshape(Up,(dims[0],dims[1],dims[2],T.M))
        Up = np.transpose(Up,[1,2,0,3])
        Up = np.reshape(Up,(dims[1]*dims[2]*dims[0],T.M))
        eigv.eigenvectors = Up
        LR = LRM.LowRank(eigv,(dims[1],dims[2],dims[0]),A)
        # Update reference timescales again:
        A.UpdateTS(LR.Timescales(A.tau))
        # Get the results:
        Ul = LR.GetComponents()
        # Update interface:
        if k == T.d-1:
            Ik = UT.ApplyLinearTransform(Y2,Ul[0],T.tensordir+"Interface%d"%(k-1))
        else:
            Ik = UT.DoubleProducts(Y2,Y3,T.tensordir+"Interface%d"%(k-1),Ul[0])
        T.SetInterface(k-1,Ik)
        # Update component k:
        T.SetComponentTensor(k,np.reshape(Ul[0].transpose(),(LR.R,dims[1],dims[2])))
        # Update next component:
        Ukplus = T.ComponentTensor(k-1)
        Uk2 = np.reshape(Ul[1].transpose(),(LR.R,dims[0],LR.M))
        Ukplus = np.einsum('ijk,lkm->ijlm',Ukplus,Uk2)
        T.SetComponentTensor(k-1,Ukplus)
        # Change the root:
        T.root = k-1
        # Store the objective function value:
        A.UpdateObjective(LR.Objective())
    return (T,A)