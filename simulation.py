import pandas as pd
import numpy as np
import itertools

from VFI import VFI_hc
from markov import discreteAR1
#################################
#   SIMULATION CLASS
#################################
class simulate:
    """
    This class delivers a simulated sample given a
    greedy policy.
    It:
        - starts with the model parameters
        - calls VFI.py to solve for the asset policy
        - interpolates the policy
        - uses it to generate an asset path for N
          observations
        - stores it as a pandas dataframe

    @author: Davide Coluccia

    --------------------------------------------------------
    INPUTS
    --------------------------------------------------------
    Parameters to be estimated
    sigma : constant elasticity of consumption
            array_like(float)
    alpha : habit in consumption persistence
            array_like(float)
    beta  : discount factor, optional default = 0.965
            array_like(float)

    rho   : persistence of income process, optional (default = 0.8)
            array_like(float)
    mu    : drift of income process, optional (default = 200)
            array_like(float)
    sigma2: variance of the income process, optional (default = 10)
            array_like(float)

    Calibrated parameters
    R     : gross interest rate, optional (default = 1.035)
            array_like(float)

    N     : Number of individuals, optional (default = 1000)
            array_like(float)
    T     : Time horizon, optional (default = 100)

    --------------------------------------------------------
    OUTPUT
    --------------------------------------------------------
    df    : simulated asset data
            pandas.DataFrame

    """
    def __init__(self, sigma, alpha, data, beta = 0.965, rho = 0.8, sigma2 = 10.0, mu = 200, R = 1.035, N = 1000, T = 100):
        self.alpha = alpha
        self.beta  = beta
        self.sigma = sigma
        self.rho   = rho
        self.sigma2= sigma2
        self.mu    = mu

        self.data  = data
        self.R     = R
        self.Pi    = np.array([])
        self.W_grid= np.array([])
        self.A_grid= np.array([])
        self.C_grid= np.array([])
        self.pol   = np.array([])
        self.df    = []

        self.N     = N
        self.T     = T

        #WorkFlow
        self.discretize_income()
        self.get_greedy()
        self.simulate_df()

    def discretize_income(self):
        'Discretize the income AR process'
        W_grid, Pi = discreteAR1(n = 10, mu = self.mu, rho = self.rho, sigma = self.sigma2, alg = 'r')

        self.W_grid = W_grid
        self.Pi = Pi

    def get_greedy(self):
        """
        Solve for the sigma-greedy policy for a given agent.
        """
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        W_grid, Pi, R = self.W_grid, self.Pi, self.R

        dp = VFI_hc(sigma, alpha, beta,
                    R, Pi, W_grid, plot = False)

        pol     = dp.Pol
        A_grid  = dp.a_grid
        C_grid  = dp.c_grid

        self.pol = pol
        self.A_grid = A_grid
        self.C_grid = C_grid

    def simulate_income(self, i):
        """
        Get from the simulated data, income for individual i.
        """
        data = self.data
        sub = data.loc[data['id'] == (i+1)]
        w = sub['income'].values.tolist()

        return w

    def one_simulate(self, i):
        """
        Simulate the asset path for a single individual.
        """
        from scipy.interpolate import interpn

        T = self.T

        W_grid = self.W_grid
        pol, A_grid, C_grid = self.pol, self.A_grid, self.C_grid

        w, c, a = self.simulate_income(i), [], []
        A_opt = A_grid[pol.astype(int)]

        c0 = w[0] ; a0 = w[0] - c0
        c.append(c0) ; a.append(a0)
        for t in range(T-1):
            # Bin down elements out of the grid
            if a0 < A_grid[0]:
                a0 = A_grid[0]
            elif a0 > A_grid[-1]:
                a0 = A_grid[-1]

            if c0 < C_grid[0]:
                c0 = C_grid[0]
            elif c0 > C_grid[-1]:
                c0 = C_grid[-1]

            a1 = interpn((A_grid, C_grid, W_grid),A_opt,(a0,c0,w[t+1]))[0]
            c1 = w[t+1] - a1

            c.append(c1) ; a.append(a1)
            a0 = a1
            c0 = c1

        return w, c, a

    def simulate_df(self):
        """
        Simulate an asset path for N individuals, and aggregate
        into a dataframe.
        """
        def itertools_chain(a):
            return list(itertools.chain.from_iterable(a))

        N = self.N; T = self.T
        ID, TT, WW, CC, AA = [], [], [], [], []

        for i in range(N):
            ii =[(i+1) for t in range(T)]
            tt =[t for t in range(T)]
            ww, cc, aa = self.one_simulate(i)

            ID.append(ii) ; TT.append(tt) ; WW.append(ww) ; CC.append(cc) ; AA.append(aa)

        ID = itertools_chain(ID) ; TT = itertools_chain(TT) ; WW = itertools_chain(WW)
        CC = itertools_chain(CC) ; AA = itertools_chain(AA)

        df_dict = {'id':ID, 'time': TT, 'income': WW, 'consumption':CC, 'assets': AA}
        df = pd.DataFrame(df_dict) ; df = df[['id', 'time', 'income', 'consumption', 'assets']]

        self.df = df

##############################
