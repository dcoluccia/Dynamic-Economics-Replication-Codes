import numpy as np
from numba import njit
from math import pi, sqrt
from scipy.stats import norm
import scipy.sparse as ss
from scipy.linalg import solve
from ctypes import POINTER, byref, cdll, c_int, c_char, c_double
from functools import partial

def discreteAR1(n, mu, rho, sigma, m=3, alg='f'):

    # Input validation
    if not isinstance(n, int):
        raise TypeError('n is not an int')
    if not isinstance(mu, (float, int)):
        raise TypeError('mu is not a number')
    if not isinstance(rho, float):
        raise TypeError('rho is not a float')
    if not isinstance(sigma, float):
        raise TypeError('sigma is not a float')
    if not isinstance(m, int):
        raise TypeError('m is not an int')
    if alg not in ('r', 't', 'th', 'f'):
        raise ValueError('Unsupported algorithm! Choose ' +
                         'r (Rouwenhorst), t (Tauchen), ' +
                         'th (Tauchen-Hussey), or f (Floden)')
    if n < 2:
        raise ValueError('n is less than 2')
    if not 0 <= rho < 1:
        raise ValueError('rho is out of range')
    if sigma <= 0:
        raise ValueError('sigma is out of range')
    if m <= 0:
        raise ValueError('m is out of range')

    # Compute std of y given std of shock
    sigma_y = sigma / sqrt(1-rho**2)

    def _rouwenhorst():

        def compute_P(p, n):

            if n == 2:
                P = sym_transition_matrix(p)
            else:
                Q = compute_P(p, n-1)
                A = np.zeros((n, n))
                B = np.zeros((n, n))
                A[:n-1, :n-1] += Q
                A[1:n, 1:n] += Q
                B[:n-1, 1:n] += Q
                B[1:n, :n-1] += Q
                P = p*A+(1-p)*B
                P[1:-1] /= 2
            return P

        p = (1+rho) / 2
        P = compute_P(p, n)
        f = sqrt(n-1)*sigma_y
        s = np.linspace(-f, f, n) + mu
        return s, P

    def _tauchen():

        s_max = m*sigma_y
        s, w = np.linspace(-s_max, s_max, n, endpoint=True, retstep=True)
        x = s - rho*s.reshape((-1, 1)) + w/2
        P = norm.cdf(x / sigma)
        P[:, -1] = 1.0
        P[:, 1:] = np.diff(P)
        s += mu
        return s, P

    def _th_core(std, std_base):

        s, w = np.polynomial.hermite.hermgauss(n)
        s *= sqrt(2)*std_base
        pdf = (norm.pdf(s, rho*s.reshape((-1, 1)), std) /
               norm.pdf(s, 0, std_base))
        P = w/sqrt(pi) * pdf
        P /= P.sum(axis=1, keepdims=True)
        s += mu
        return s, P

    def _tauchen_hussey():

        return _th_core(sigma, sigma)

    def _floden():

        w = 1/2 + rho/4
        sigma_hat = w*sigma + (1-w)*sigma_y
        return _th_core(sigma, sigma_hat)

    switcher = {'r': _rouwenhorst,
                't': _tauchen,
                'th': _tauchen_hussey,
                'f': _floden}

    return switcher[alg]()
