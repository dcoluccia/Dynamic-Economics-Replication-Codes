import numpy as np
import matplotlib.pyplot as plt

from time   import time
from numba  import jit

#################################
#   CLASS
#################################
class VFI_hc:
    """
    --------------------------------------------------------
    GENERAL DESCRIPTION
    --------------------------------------------------------
    This class solves the household intertemporal problem
    when there are habit in consumption.

    The problem is
        V(a) = max U(ctilde) + beta E[V(a')]
    s.t.
        c + a' = R*a + W
        c_tilde = c - alpha*cc

    W is random. W is AR(1), and discretized as Markov.
    The state-space is (a,cc), where cc is past consumption.

    @author: Davide Coluccia
    
    --------------------------------------------------------
    INPUTS
    --------------------------------------------------------
    sigma : constant elasticity of consumption
            array_like(float)
    alpha : habit in consumption persistence
            array_like(float)
    beta  : discount factor
            array_like(float)
    R     : gross interest rate
            array_like(float)
    Pi    : transition probability of income
            array_like(np.array)
    W_grid: grid for income values
            array_like(np.array)
    
    plot  : whether to plot simulated Value Function and Policy
            optional, default == False
            bool: True/False
    dimA  : dimension of the asset grid
            optional, default == 1000
            array_lile()
            
    --------------------------------------------------------
    OUTPUT
    --------------------------------------------------------
    pol   : policy function
            array_like(np.array)
    
    V1    : value function
            array_like(np.array)
            
    """
    def __init__(self, sigma, alpha, beta, R, Pi, W_grid, plot = False, dimA = 10):
        self.sigma  = sigma
        self.alpha  = alpha
        self.beta   = beta
        self.R      = R
        self.Pi     = Pi
        self.W_grid = W_grid
        
        self.dimA   = dimA
        self.dimW   = np.shape(Pi)[0]
        
        self.tol = 0
        self.Niter = 0
        self.a_grid = np.array([])
        self.c_grid = np.array([])
        
        self.V1  = np.array([])
        self.Pol = np.array([])
        
        # WorkFlow
        self.grid()
        self.solve_VFI()
        if plot == True:
            self.plot_graph()
        
    @jit  
    def u(self,c,x):
        """
        Returns utility given current consumption c, and past 
        consumption x.
            U(c,x) = ctilde^(1-sigma) / (1-sigma)
        where
            ctilde = c - alpha*x
        """
        alpha = self.alpha ; sigma = self.sigma
        
        ctilde = c - alpha*x
        u = ctilde**(1-sigma) / (1-sigma)
        
        return u

    def grid(self):
        """
        - Define grids for assets, past consumption and income;
        - Define tolerance and maximum iteration fro VFI algorithm.
        """
        dimA = self.dimA ; dimC = self.dimA ; W_grid = self.W_grid
        
        self.tol = 10e-5
        self.Niter = 10000
        
        a0 = 100 / self.dimA
        c0 = 100 / self.dimA
        a_grid = np.mgrid[0:(dimA):1] ; a_grid = a0 * a_grid ; self.a_grid = a_grid
        c_grid = np.mgrid[0:(dimC):1] ; c_grid = c0 * c_grid ; self.c_grid = c_grid
        self.W_grid = W_grid
    
    def solve_VFI(self):
        """
        Solve the problem of the household using value function iteration.
        Note: in the future:
            - vectorize;
            - possibly, solve for finite horizon by backward induction.
        """
        dimC = self.dimA ; dimA = self.dimA ; dimW = self.dimW 
        C = self.c_grid ; A = self.a_grid ; W = self.W_grid
        tol = self.tol ; Niter = self.Niter ; R = self.R
        beta = self.beta ; Pi = self.Pi
        
        V0  = np.zeros((dimA,dimC,dimW))
        V1  = np.zeros((dimA,dimC,dimW))
        Pol = np.zeros((dimA,dimC,dimW))
        U   = np.zeros((dimA,dimC,dimW))
        
        t0 = time()
        diff = 1 ; niter = 0
        
        while diff > tol:
            niter += 1
            # Value update step
            for ia in range(dimA):
                for ic in range(dimC):
                    for iw in range(dimW):
                        c = W[iw] + R*A[ia] - A
                        x = C[ic]
                        
                        c[c < 0] = np.nan 
                        if x < 0:
                            x = np.nan
                            
                        u = self.u(c,x) 
                        U[:,ic,iw] = u   
                        
                Objective   = U + beta * V0 @ Pi.T
                V1[ia,:,:]  = np.nanmax(Objective, axis = 0)
                Pol[ia,:,:] = np.nanargmax(Objective, axis = 0)
            
            # Evaluate distance between the value functions
            diff = np.max(np.max(np.abs(V1 - V0))) 
            V0[:] = V1
            
            # Break the while loop if too many iterations
            #print("The current error is "+str(diff))
            if niter > Niter:
                print('Ops, no convergence')
                break
    
        t1 = time()
        #print('VFI algorithm took {0:0d} iterations and {1:.2f} seconds.'.format(niter, t1 - t0))
        
        self.V1 = V1 ; self.Pol = Pol
    
    def plot_graph(self):
        """
        If "plot" option == True, this function is executed, and delivers some plots.
        """
        A = self.a_grid ; V = self.V1 ; Pol = self.Pol
        A_opt = A[Pol.astype(int)]
        
        fig = plt.subplots(figsize = (8,5))
        ax = [None,None]
        pltgrid = (1,2)
        
        ax[0] = plt.subplot2grid(pltgrid, (0,0))
        ax[1] = plt.subplot2grid(pltgrid, (0,1))
        
        ax[0].plot(A[:],V[:,0,0], linewidth = 2, color = 'blue', label = r'$V(a)$: Low $w$')
        ax[0].plot(A[:],V[:,0,5], linewidth = 2, color = 'green', label = r'$V(a)$: Median $w$')
        ax[0].plot(A[:],V[:,0,-1], linewidth = 2, color = 'red', label = r'$V(a)$: High $w$')
        
        ax[1].plot(A[:],A_opt[:,0,0], linewidth = 2, color = 'blue', label = r'$a\'(a)$: Low $w$')
        ax[1].plot(A[:],A_opt[:,0,5], linewidth = 2, color = 'green', label = r'$a\'(a)$: Median $w$')
        ax[1].plot(A[:],A_opt[:,0,-1], linewidth = 2, color = 'red', label = r'$a\'(a)$: High $w$')
        ax[1].plot(A[:],A[:], linewidth = 2, color = 'violet', linestyle = 'dashed', zorder = 1)
        
        
        ax[0].set_xlabel(r'$a$') ; ax[0].legend()
        ax[1].set_xlabel(r'$a$') ; ax[1].legend()
        ax[0].set_title('Value function')
        ax[1].set_title('Asset policy')
        
        plt.tight_layout()
        plt.show()                    
        
###########################
#alpha0 = 0.2
#beta0  = 0.965
#sigma0 = 1.5
#R = 1.035
#W_grid, Pi = discreteAR1(n = 10, mu = 200, rho = 0.8, sigma = 10.0, alg = 'r')
#
#res = VFI_hc(sigma0,alpha0,beta0,R,Pi,W_grid,plot=True)