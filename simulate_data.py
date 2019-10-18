import pandas as pd
import numpy  as np
import itertools

from VFI import VFI_hc
from markov import discreteAR1
#################################
#   Parameters of the DGP
#################################
N       = 1000
T       = 100

rho     = 0.800
sigma2  = 10.00
alpha   = 0.200
beta    = 0.965
sigma   = 0.850
R       = 1.035
#################################
#   CLASS
#################################
class Simulate_data:
    """
    This program populates a dataframe with simulated data.
    Data on:
        W : wage
        C : consumption
        A : assets = W - C

    Structure of the output .csv data is:
        ===============================================
        id   |   Time   |    W    |    C    |    A    |
        ===============================================
        1    |     0    |   100   |    80   |    20   |
        1    |     1    |   102   |    81   |    21   |
        1    |     .    |    .    |     .   |     .   |
        1    |     .    |    .    |     .   |     .   |
        1    |     .    |    .    |     .   |     .   |
        1    |     T    |  120    |    99   |    21   |
        -----------------------------------------------
        2    |     0    |   100   |    80   |    20   |
        2    |     1    |   102   |    81   |    21   |
        2    |     .    |    .    |     .   |     .   |
        2    |     .    |    .    |     .   |     .   |
        2    |     .    |    .    |     .   |     .   |
        2    |     T    |  120    |    99   |    21   |
        -----------------------------------------------
        .    |     .    |    .    |     .   |     .   |
        .    |     .    |    .    |     .   |     .   |
        .    |     .    |    .    |     .   |     .   |
        -----------------------------------------------
        N    |     0    |   100   |    80   |    20   |
        N    |     1    |   102   |    81   |    21   |
        N    |     .    |    .    |     .   |     .   |
        N    |     .    |    .    |     .   |     .   |
        N    |     .    |    .    |     .   |     .   |
        N    |     T    |  120    |    99   |    21   |
        ===============================================
    where
        N : number of individuals
        T : number of periods

    The DGP for assets and consumption is assumed to be the structural model.
    Income is exogenous:
        W_t = rho*W_{t-1} + epsilon
    with epsilon N(0,sigma2).

    @author: Davide Coluccia
    """
    def __init__(self, N, T, rho, sigma2, alpha, beta, sigma, R, mu = 200.0, save = True):
        self.N      = N
        self.T      = T

        self.rho    = rho
        self.sigma2 = sigma2
        self.mu     = mu

        self.alpha  = alpha
        self.beta   = beta
        self.sigma  = sigma
        self.R      = R

        self.pol    = np.array([])
        self.df     = np.array([])
        self.W_grid = np.array([])
        self.Pi     = np.array([])
        self.A_grid = np.array([])
        self.C_grid = np.array([])

        #WorkFlow
        self.discretize_income()
        self.get_greedy()
        self.simulate_df()
        if save == True:
            self.store()

    def _itertools_chain(a):
        return list(itertools.chain.from_iterable(a))

    def discretize_income(self):
        'Discretize the income AR process'
        W_grid, Pi = discreteAR1(n = 10, mu = self.mu, rho = self.rho, sigma = self.sigma2, alg = 'r')

        self.W_grid = W_grid
        self.Pi = Pi

    def get_greedy(self):
        'Get the greedy policy for assets'
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        W_grid, Pi, R = self.W_grid, self.Pi, self.R

        dp = VFI_hc(sigma, alpha, beta,
                    R, Pi, W_grid)

        pol     = dp.Pol
        A_grid  = dp.a_grid
        C_grid  = dp.c_grid

        self.pol = pol
        self.A_grid = A_grid
        self.C_grid = C_grid

    def simulate_income(self):
        """
        Interpolate the income process for a given individual.
        """
        Pi, W_grid = self.Pi, self.W_grid

        w = []

        w0 = np.random.choice(W_grid, size = 1).tolist()[0] ;
        w.append(w0)

        i0 = W_grid.tolist().index(w0)

        for t in range(self.T-1):
            w1 = np.random.choice(W_grid, size = 1, p = Pi[i0,:]).tolist()[0] ; w.append(w1)
            i1 = W_grid.tolist().index(w1)

            i0 = i1

        return w

    def one_simulate(self):
        """
        Simulate the asset path for a single individual.
        """
        from scipy.interpolate import interpn

        T = self.T

        W_grid = self.W_grid
        pol, A_grid, C_grid = self.pol, self.A_grid, self.C_grid

        w, c, a = self.simulate_income(), [], []
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
            ww, cc, aa = self.one_simulate()

            ID.append(ii) ; TT.append(tt) ; WW.append(ww) ; CC.append(cc) ; AA.append(aa)

        ID = itertools_chain(ID) ; TT = itertools_chain(TT) ; WW = itertools_chain(WW)
        CC = itertools_chain(CC) ; AA = itertools_chain(AA)

        df_dict = {'id':ID, 'time': TT, 'income': WW, 'consumption':CC, 'assets': AA}
        df = pd.DataFrame(df_dict) ; df = df[['id', 'time', 'income', 'consumption', 'assets']]

        self.df = df

    def store(self):
        'Save the dataframe'

        self.df.to_csv('simulated_data.csv', index = False)

#################################
#   RUN
#################################
sample = Simulate_data(N,T,rho,sigma2,alpha,beta,sigma,R, save = True)
