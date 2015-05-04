class ALS:
    def __init__(self,T,k):
        ''' This is just a container that keeps track of relevant quantities during
        a call to ALS.
        
        Parameters:
        -------------
        T: BlockTTtensor object, represents the TT-tensor to be optimized.
        k: int, number of eigenvalues requested.
        '''
        self.T = T
        self.k = k
        # Determine the order of the tensor:
        self.d = self.T.d
    def RunALS(self,eps_iter=1e-2,eps_orth=1e-8,eps_rank=0.9,mu0=10):
        ''' Run the ALS scheme on the tensor T.
        
        Parameters:
        ------------
        eps: float,convergence criterion, stop iteration if objective value did not
            differ by more than eps.
        tol, float, tolerance for rank-control during tensor-reshape.
        tol2: float, tolerance for orthogonality of solutions.
        rtol: int, minimal rank for optimization attempt.
        
        Returns:
        ------------
        D: ndarray, SHAPE (self.k,2*Niter), contains all eigenvalues after every half-sweep.
            self.k is the number of eigenvalues, Niter is the number of iterations that were
            performed by the method.
        Efun: ndarray, SHAPE (self.Tlen,self.k), each column corresponds to the evaluation of one
            eigenfunction along the full trajectory.
        '''
        print "Starting Iterative Procedure"
        # Prepare output for eigenvalues:
        D = np.zeros(0)
        # References for timescales:
        tsmax = np.zeros(self.k-1)
        # Start the iteration:
        q = 0
        J = 0
        while J == 0:
            print "-------------"
            print "Iteration %d"%(q+1)
            print "-------------"
            # Perform one full sweep:
            Dq,tsmax = self.ALSSweep(eps_orth,eps_rank,mu0,tsmax,analyse=1,sweep_id=q)
            # Store the results:
            D = np.hstack((D,Dq))
            # Get objective value:
            Jq = Dq[-1]
            # Check for convergence:
            if np.abs(Jq - J) < eps_iter:
                break
            else:
                J = Jq
                q += 1
        # Evaluate the eigenfunctions:
        Efun = self.T.EvaluateFullTensor(self.tau,self.sep)
        # Return the eigenvalues:
        return (D,Efun)
    def ALSSweep(self,eps_orth,eps_rank,mu0,tsmax,analyse = 0,sweep_id=0):
        ''' Perform one up-and-down sweep through the tensor.
        
        Parameters:
        ------------
        eps_orth, eps_rank: float, tolerances for orthogonality of solutions,
            rank control.
        mu0: float, starting value for penalty algorithm.
        tsmax: nd-array, shape(self.k-1,), array of reference timescales for
            rank control.
        analyse: bool, perform component analysis during iteration.
        sweep_id: int, number of the current sweep, only needed if analyse=True. 
        
        Returns:
        ------------
        evs: ndarray, SHAPE (self.k,ni), contains values of the objective func-
            tion after each iteration step.
        tsmax: nd-array, shape(self.k-1,), array of reference timescales for
            rank control, updated.
        '''
        # Prepare output:
        evs = np.zeros(2*(self.d-1))
        # Initialize the evaluation array:
        Y1 = np.ones((self.Tlen,1))
        # Prepare list of objectives for analysis:
        obj_array = np.zeros(2*(self.d-1))
        # Loop forward over the tensor:
        for k in range(self.d-1):
            print "Forward problem %d"%(k)
            rkm = np.shape(Y1)[1]
            # Get the evaluation of this coordinate basis:
            fk = self.T.ComponentTimeSeries(k)
            nk = np.shape(fk)[1]
            # Load the right evaluation:
            Y2 = np.load(self.T.EvaluationFile(k+1))
            rkp = np.shape(Y2)[1]
            # Compute the triple products:
            TripleEval = RH.ComputeTripleProducts(Y2,Y1,fk)
            # Call the eigenvalue solver:
            Vk,Dk,Ctau,C0 = RH.RHEigenvalue(TripleEval, self.k, self.tau, self.sep)
            # Perform additional analysis if requested:
            if analyse:
                # Compute objective function without this coordinate:
                obj_array[k] = self.LeaveOutComponent(Y1,Y2)
            # Update reference timescales:
            ts_k = -self.tau/np.log(Dk[:-1])
            tsmax = np.maximum(tsmax,ts_k)
            print "Eigenvalues:"
            print Dk
            # Perform low-rank procedure:
            Ukp = Vk.transpose()
            Uk,Wk,rnew,Ofun,tsmax = self.LowRank(Ukp,rkp,rkm,nk,Ctau,C0,eps_orth,eps_rank,mu0,tsmax)
            Uk = Uk.transpose()
            Wk = Wk.transpose()
            # Compute products of Y1 and fk:
            Y1 = RH.ComputeDoubleProducts(Y1,fk)
            # Compute new evaluations:
            Y1 = np.dot(Y1,Wk)
            # Save the result:
            np.save(self.tensordir + "Interface_%d.npy"%k,Y1)
            # Update evaluations:
            self.T.SetEvaluationFile(k,self.tensordir + "Interface_%d.npy"%k)
            self.T.SetEvaluationFile(k+1,"")
            # Replace component k by Uk:
            Wk = np.reshape(Wk,(rkm,nk,rnew))
            self.T.SetComponentTensor(k,Wk)
            # Update next component:
            Uk = np.reshape(Uk,(rnew,self.k,rkp))
            Ukplus = self.T.ComponentTensor(k+1)
            Uk = np.einsum('ijk,klm->ilmj',Uk,Ukplus)
            self.T.SetComponentTensor(k+1,Uk)
            # Change the root:
            self.T.SetRoot(k+1)
            # Store the objective function value:
            evs[k] = Ofun
        # When the forward sweep is over, re-initialize Y2:
        Y2 = np.ones((self.Tlen,1))
        # Prepare an array for the interface-contributions:
        CIf = np.zeros(self.d-1)
        # Loop backward over the tensor:
        for k in range(self.d-1,0,-1):
            print "Backward problem %d"%(k)
            rkp = np.shape(Y2)[1]
            # Get the evaluation of this coordinate basis:
            fk = self.T.ComponentTimeSeries(k)
            nk = np.shape(fk)[1]
            # Load the right evaluation:
            Y1 = np.load(self.T.EvaluationFile(k-1))
            rkm = np.shape(Y1)[1]
            # Compute the triple products:
            TripleEval = RH.ComputeTripleProducts(Y1, fk, Y2)
            # Call the eigenvalue solver:
            Vk,Dk,Ctau,C0 = RH.RHEigenvalue(TripleEval, self.k, self.tau, self.sep)
            # Perform additional analysis if requested:
            if analyse:
                # Compute objective function without this coordinate:
                obj_array[2*(self.d-1) - k] = self.LeaveOutComponent(Y1,Y2)
            ts_k = -self.tau/np.log(Dk[:-1])
            tsmax = np.maximum(tsmax,ts_k)
            print "Eigenvalues:"
            print Dk
            # Compute low-rank decomposition:
            Ukp = Vk.transpose()
            Uk,Wk,rnew,Ofun,tsmax = self.LowRank(Ukp,rkm,nk,rkp,Ctau,C0,eps_orth,eps_rank,mu0,tsmax)
            # Compute product of fk and Y2
            Y2 = RH.ComputeDoubleProducts(fk,Y2)
            # Evaluate the new interface:
            Y2 = np.dot(Y2,Wk.transpose())
            # Save the result:
            np.save(self.tensordir + "Interface_%d.npy"%k,Y2)
            # Also compute contribution from interface:
            _,DY2,_,_ = RH.RHEigenvalue(Y2,self.k,self.tau,self.sep)
            CIf[k-1] = -0.5*np.sum(DY2)
            # Update evaluations:
            self.T.SetEvaluationFile(k,self.tensordir + "Interface_%d.npy"%k)
            self.T.SetEvaluationFile(k-1,"")
            # Replace component k by Wk:
            Wk = np.reshape(Wk,(rnew,nk,rkp))
            self.T.SetComponentTensor(k,Wk)
            # Update next component:
            Uk = np.reshape(Uk,(self.k,rkm,rnew))
            Ukminus = self.T.ComponentTensor(k-1)
            Uk = np.einsum('ijk,mkn->ijnm',Ukminus,Uk)
            self.T.SetComponentTensor(k-1,Uk)
            # Change the root:
            self.T.SetRoot(k-1)
            # Store the objective function value:
            evs[2*(self.d-1) - k] = Ofun
        # Save the interface contributions:
        np.savetxt(self.tensordir + "Interface_Objectives.dat",CIf)
        # Save the objectives without components:
        np.savetxt(self.tensordir + "Objectives_NoComponents_Run%d.dat"%sweep_id,obj_array)
        return (evs,tsmax)
    def LowRank(self,Ukp,n1,n2,n3,Ctau,C0,eps_orth,eps_rank,mu0,tsmax):
        ''' Computes optimal low-rank decomposition of the full solution for the
        subproblem in ALS.
        
        Parameters:
        --------------
        Ukp: ndarray, shape(self.k,n1*n2*n3), the solution of the subproblem.
        n1,n2,n3: float, the dimensions indicated above.
        Ctau, C0: ndarray, shape(n1*n2*n3,n1*n2*n3), the correlation matrices
            of the basis functions.
        eps_orth, eps_rank: float, tolerances for orthogonality of solutions,
            rank control.
        mu0: float, starting value for penalty algorithm.
        tsmax: nd-array, shape(self.k-1,), array of reference timescales for
            rank control.
        
        Returns:
        --------------
        Uk,Wk: ndarray, shape(self.k*n1,rnew) and shape(rnew,n2*n3): The components
            of the low-rank decomposition.
        rnew: float, the new separation rank.
        Ofun: float, the objective function from the low-rank solution.
        tsmax: nd-array, shape(self.k-1,), updated array of reference timescales.  
        '''
        # Compute the eigenvalue trace:
        ktrace = -0.5*np.trace(np.dot(Ukp,np.dot(Ctau,Ukp.transpose())))
        # Reshape Ukp to desired shape:
        Ukp = np.reshape(Ukp,(self.k*n1,n2*n3))
        # Compute initial low-rank decomposition:
        Uk,sk,Wk = scl.svd(Ukp,full_matrices=False)
        rfull = np.shape(sk)[0]
        Wk = np.dot(np.diag(sk),Wk)
        # Start adaptive rank selection:
        rnew = 1
        while 1:
            # Select only the first rnew singular values:
            Ukr = Uk[:,:rnew]
            Wkr = Wk[:rnew,:]
            # Compute truncated Ukp:
            Ukp = np.dot(Ukr,Wkr)
            # Reshape for evaluation:
            Ukp2 = np.reshape(Ukp,(self.k,n1*n2*n3))
            # Check for orthonormality of solutions:
            Unorm = np.dot(Ukp2,np.dot(C0,Ukp2.transpose()))
            rorth = np.amax(np.abs(Unorm - np.eye(self.k,self.k)))
            Ofun = -0.5*np.trace(np.dot(Ukp2,np.dot(Ctau,Ukp2.transpose())))
            # Check if orthogonality constraints are not violated:
            if rorth < eps_orth:
                print "Solutions are orthonormal."
            # Otherwise, try to compute a sufficiently orthonormal solution
            # by optimization:
            else:
                print "Attempting Optimization."
                U0 = CT.CanonicalTensor(rnew,np.shape(Ukp)[0],np.shape(Ukp)[1])
                for i in range(rnew):
                    U0.SetComponent(0,i,Ukr[:,i])
                    U0.SetComponent(1,i,Wkr[i,:])
                Lambda = np.ones(0.5*self.k*(self.k+1))
                eps_array = eps_orth*np.ones(0.5*self.k*(self.k+1))
                TUR, Ofun = CGM.PenaltyCG(U0,Ctau,C0,self.k,eps_array,ktrace,Lambda,mu0)
                Ukr = TUR.GetDim(0).transpose()
                Wkr = TUR.GetDim(1)
                Ukp2 = TUR.GetVU()
                Ukp2 = np.reshape(Ukp2,(self.k,n1*n2*n3))
            if Ofun == 0:
                print "Low-Rank approximation failed for rnew = %d."%rnew
                rnew += 1
                continue
            # Check if implied timescales are sufficiently restored:
            # Compute correlation matrix of the solutions:
            Ctau_U = np.dot(Ukp2,np.dot(Ctau,Ukp2.transpose()))
            C0_U = np.dot(Ukp2,np.dot(C0,Ukp2.transpose()))
            # Diagonalize it:
            DU,_ = scl.eigh(Ctau_U,C0_U)
            ind = np.argsort(DU)
            DU = DU[ind]
            ts_k = -self.tau/np.log(DU[:-1])
            print "Current Timescales:"
            print ts_k
            print "Reference timescales:"
            print tsmax
            # Compare timescales:
            if np.all(ts_k >= eps_rank*tsmax) or rnew==rfull:
                tsmax = np.maximum(tsmax,ts_k)
                Uk = Ukr
                Wk = Wkr
                print "New objective value: %.5e"%Ofun
                break
            else:
                rnew += 1
        print "Rank modified to %d"%rnew
        print ""
        return (Uk,Wk,rnew,Ofun,tsmax)
    def LeaveOutComponent(self,Yl,Yr):
        ''' Solve optimization problem without using the coordinate basis for
        a specific coordinate.
        
        Parameters:
        ------------
        Yl: ndarray, shape(self.Tlen,r_p-1), the left interface time series.
        Yr: ndarray, shape(self.Tlen,r_p), the right interface time series.
        Returns:
        ------------
        obj: float, the objective function from solving this ev-problem.
        '''
        # Compute double products:
        Yd = RH.ComputeDoubleProducts(Yl,Yr)
        if Yd.shape[1] == 1:
            return -0.5
        else:
            # Solve eigenvalue problem:
            _,D,_,_ = RH.RHEigenvalue(Yd,np.minimum(self.k,Yd.shape[1]),self.tau,self.sep)
            # Compute objective function and return:
            return -0.5*np.sum(D)