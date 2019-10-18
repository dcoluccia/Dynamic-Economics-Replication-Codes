import numpy as np
from numba import njit
from math import pi, sqrt
from scipy.stats import norm
import scipy.sparse as ss
from scipy.linalg import solve
from ctypes import POINTER, byref, cdll, c_int, c_char, c_double
from functools import partial


def validate_transition_matrix(P):
    """
    Sanity check on transition matrices
    """
    if ss.issparse(P):
        P = P.tocsr()
    else:
        P = np.atleast_2d(P)
    if not P.ndim == 2:
        raise TypeError('P is not a 2D array')
    n, m = P.shape
    if not n == m:
        raise TypeError('P is not square')
    if n < 2:
        raise ValueError('At least two states are needed')
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError('P is not right stochastic')
    return P


def validate_state_vector(s, P):
    """
    Sanity check on state vectors
    """
    s = np.atleast_2d(s)
    if s.ndim > 2:
        raise TypeError('s has to be at most 2D')
    if not s.size/s.shape[0] == P.shape[0]:
        raise ValueError('s and P do not conform!')
    return s.squeeze()


def ergodic_dist_mc(Q, t=10000):

    if ss.issparse(Q):
        Q = Q.todense()
    cQ = Q.cumsum(axis=1)
    return _simulate_ergodic(cQ, t)


@njit(cache=True)
def _simulate_ergodic(cQ, t):

    n = cQ.shape[0]
    q = np.zeros(n, dtype=np.int64)
    past = 0
    for i in range(t):
        j = 0
        draw = np.random.random_sample()
        while cQ[past, j] < draw:
            j += 1
        q[j] += 1
        past = j
    return q/t


def ergodic_dist(Q, alg='dir', tol=1e-6):
    """
    Ergodic distribution of a discrete Markov Chain

    q = ergodicdist(Q, alg='dir', tol=1e-6, validate=True)

    Input:  Q - right stochastic 2D numpy array or sparse matrix.
            alg - choice of method:
                     'dir': direct (default),
                     'iter': iterative BICGSTAB,
                     'power': power method (direct call to MKL)
    Output: q - 1D dense numpy array containing the ergodic distribution

    Author: Marco Maffezzoli. Ver. 3.1, 11/2017.
    """
    Q = validate_transition_matrix(Q)
    if alg not in ('dir', 'iter', 'power'):
        raise ValueError('Method not implemented')
    if not isinstance(tol, float):
        raise TypeError('tol is not a float')

    # Initialization
    def _matrices(Q):

        n, m = Q.shape
        if ss.issparse(Q):
            A = ss.eye(n, format='csr')
            A -= Q.tocsr().transpose()
            index = (np.ones(n), (np.zeros(n), np.arange(n)))
            A += ss.csr_matrix(index, (n, m))
        else:
            A = np.eye(n)
            A -= Q.transpose()
            A[0, ] += np.ones(n)
        b = np.zeros(n)
        b[0] += 1.0
        return A, b

    # Direct method
    def _direct(Q):

        A, b = _matrices(Q)
        if ss.issparse(A):
            return ss.linalg.spsolve(A, b)
        else:
            return solve(A, b, overwrite_a=True, overwrite_b=True)

    # Iterative method
    def _iterative(Q):

        A, b = _matrices(Q)
        n = b.size
        q = np.full(n, 1/n)
        q, info = ss.linalg.lgmres(A, b, q)
        if info > 0:
            raise RuntimeError('Convergence not achieved.')
        elif info < 0:
            raise RuntimeError('Illegal input or breakdown.')
        else:
            return q

    # Power method (with direct call to MKL)
    def _power(Q):

        n = Q.shape[0]
        q = np.full(n, 1/n)
        diff = 1.0
        if ss.issparse(Q):
            z = np.zeros_like(q)
            mkl = cdll.LoadLibrary('mkl_rt.dll').mkl_cspblas_dcsrgemv
            data = Q.data.ctypes.data_as(POINTER(c_double))
            indp = Q.indptr.ctypes.data_as(POINTER(c_int))
            indi = Q.indices.ctypes.data_as(POINTER(c_int))
            transa = byref(c_char(ord('T')))  # Q is transposed
            nrows = byref(c_int(n))
            spmv = partial(mkl, transa, nrows, data, indp, indi)
            qc = q.ctypes.data_as(POINTER(c_double))
            zc = z.ctypes.data_as(POINTER(c_double))
            while diff > tol:
                spmv(qc, zc)
                spmv(zc, qc)
                diff = np.linalg.norm(z-q)
        else:
            while diff > tol:
                z = q.dot(Q)
                q = z.dot(Q)
                diff = np.linalg.norm(z-q)
        return q

    return {'dir': _direct,
            'iter': _iterative,
            'power': _power}[alg](Q)


def sym_transition_matrix(p):
    """
    Returns a symmetric bivariate transition matrix
    """
    assert 0 <= p <= 1
    return np.array([[p, 1-p], [1-p, p]], dtype=np.float64)


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
