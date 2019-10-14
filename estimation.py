import numpy as np

from simulation import simulate
from numba import jit
from time import time
#################################
#   ESTIMATION CLASS
#################################
class simulated_mm:
    """
    This class takes as input the raw data.
    It:
        - evaluates the sample moments of the data;
        - guess theta -> call simulate.py that: 
            solve VFI -> simulates data 
        - evaluates the moments of the simulated data
        - compare them with the actual ones and repeat
    
    @author: Davide Coluccia
    
    --------------------------------------------------------
    INPUTS
    --------------------------------------------------------
    Estimation parameter initial guess
    
    sigma0 : constant elasticity of consumption
             array_like(float)
    alpha0 : habit in consumption persistence
             array_like(float)
    beta0  : discount factor
             array_like(float)
    
    SMM estimation parameters 
        
    data   : dataframe of raw data 
             pandas.DataFrame
            
    --------------------------------------------------------
    OUTPUT
    --------------------------------------------------------
    est    : estimates of the parameters
             array_like(np.array)
    est_se : var/covar matric of the estimates
             array_like(np.array)
             NOTE: STANDARD ERRORS NEED TO BE IMPLEMENTED
            
    """
    def __init__(self, sigma0, alpha0, data):
        self.sigma0 = sigma0
        self.alpha0 = alpha0
        
        self.data  = data
        
        self.data_moments   = np.array([])
        self.est_coef       = np.array([])
        self.est_se         = np.array([])
        
        self.alpha_grid = np.array([])
        self.sigma_grid = np.array([])
        self.outcome_function = np.array([])
        #WorkFlow
        self.data_moments = self.compute_moments(self.data)
        #self.run_estimation() 
        self.brute_run_estimation(double = True)
        #self.get_estimates(double = False)
    
    @jit
    def autocovariance(self, Xi, k):
        'Compute the k-autocovariance of Xi.'
        N = np.size(Xi) ; Xs = np.mean(Xi)
        autoCov = 0
        for i in range(N-k):
            autoCov += (Xi[i+k]-Xs)*(Xi[i]-Xs)
        return autoCov / (N-1)
    
    def compute_moments(self, data):
        """
        Compute the moments of the input. Input must be
        a pandas.DataFrame objects.
        By default, moments are 
            - Mean
            - Variance
            - 1 and 2 order autocovariance
            
        Moments are computes as between-individual averages
        of within-individual moments.
        """
        from  scipy.stats import skew, kurtosis
        
        moments = np.array([])
        T = len(data.loc[data['id'] == 1])
        N = len(list(dict.fromkeys(data['id'].values.tolist())))
        
        mean = [] ; var = [] ; cov1 = [] ; cov2 = [] ; cov3 = [] ; sskew = [] ; kurt = []
        for i in range(N):
            subsample = data.loc[data['id'] == (i+1)]
            X = subsample['assets'].values.tolist()
        
            mean.append(np.average(X))
            var.append(np.var(X))
            cov1.append(self.autocovariance(X,1))
            cov2.append(self.autocovariance(X,2))
            cov3.append(self.autocovariance(X,3))
            sskew.append(skew(X))
            kurt.append(kurtosis(X))
            
        mean = np.sum(mean)/N ; var = np.sum(var)/N 
        cov1 = np.sum(cov1)/N ; cov2 = np.sum(cov2)/N ; cov3 = np.sum(cov3)/N
        sskew = np.sum(sskew)/N ; kurt = np.sum(kurt)/N
        moments = np.array([mean, var, cov1, cov2])#, cov3, sskew, kurt])
       
        return moments
        
    def one_step_estimation(self, params):
        """
        Takes as input the data moments. Starts with a guess
        for the parameters, calls the simulation.py class to 
        simulate the resulting series from the policy, calls the
        compute_moments function to compute the moments of the 
        simulated dataset, and then runs GMM.
        """
        #alpha, beta, sigma = params[0], params[1], params[2]
        alpha, sigma = params[0], params[1]
        
        simulation = simulate(sigma, alpha, self.data)
        simulated_data = simulation.df
        
        actual_moments = self.data_moments
        simulated_moments = self.compute_moments(simulated_data)
        
        diff  = simulated_moments - actual_moments ; d = np.size(diff)
        print(diff)
        weight = np.array([0.5,0.5,0.5,0.5])# np.array([0.1,0.1,0.8,0.5])
        weight_matrix = np.eye(d)#np.diag(weight)
        score = diff.T @ weight_matrix @ diff
        
        return score
        
    def run_estimation(self):
        """
        Minimize one_step_estimation over a parameter grid to
        obtain GMM estimates.
        """
        from scipy.optimize import minimize
        
        alpha0, sigma0 = self.alpha0, self.sigma0
        #x0 = [alpha0, beta0, sigma0]
        x0 = [alpha0, sigma0]
        
        #bnds = ((0.0,0.5),(0.8,0.999),(0.3,0.9))
        bnds = ((0.1,0.5), (0.1,0.9))
        res = minimize(self.one_step_estimation,
                       x0 = x0, method = 'COBYLA',
                       bounds = bnds, options = {'verbose' : 1})
        
        self.est_coef = res
    
    def brute_run_estimation(self, double):
        """
        Define a grid to get around the implementation problems
        due to minimization algorithm.
        """
        alpha_grid = np.mgrid[0.0:0.3:0.05] ; dimA = np.size(alpha_grid)
        #alpha_grid = np.mgrid[-0.5:0.5:0.1] ; dimA = np.size(alpha_grid)
        sigma_grid = np.mgrid[0.350:1.150:0.1] ; dimS = np.size(sigma_grid)
        
        t0 = time() ; niter = 0
        
        if double == True:
            fun = np.zeros((dimA,dimS))
            for ialpha in alpha_grid:
                for isigma in sigma_grid:
                    niter += 1
                    
                    index_alpha = np.where(alpha_grid == ialpha)
                    index_sigma = np.where(sigma_grid == isigma)
                    
                    params = [ialpha, isigma]
                    fun[index_alpha, index_sigma] = self.one_step_estimation(params)
                    
                    print('Do not worry, we are still at iteration number '+str(niter))
        elif double == False:
            fun = np.zeros((dimA))
            for ialpha in alpha_grid:
                niter += 1
                index_alpha = np.where(alpha_grid == ialpha)
                params = [ialpha, 0.850]
                fun[index_alpha] = self.one_step_estimation(params)
                
                print('Do not worry, we are still at iteration number '+str(niter))
        t1 = time()
        print('The minimization algorithm took {0:0d} iterations and {1:.2f} seconds.'.format(niter, t1 - t0))
                
        self.alpha_grid = alpha_grid
        self.sigma_grid = sigma_grid
        self.outcome_function = fun 
        
    def get_estimates(self, double):
        """
        Get an estimate for the coefficients given the optimization routine
        together with the SE.
        """
        objective       = self.outcome_function
        min_function    = objective.min()
        if double == True:
            alpha_index, sigma_index = np.where(objective == min_function)
            alpha_sol, sigma_sol = self.alpha_grid[alpha_index] , self.sigma_grid[sigma_index]
            
            res = [alpha_sol, sigma_sol]
        elif double == False:
            alpha_index = np.where(objective == min_function)
            alpha_sol = self.alpha_grid[alpha_index]
            
            res = alpha_sol
            
        self.est_coef   = res        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        