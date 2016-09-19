import numpy as np
import scipy.signal


def stm(y, x, NZ, NP, N):
    ''' steiglitz-mcbride N iterations, NZ zeros, NP poles
    input x(n), output y(n), returns filter coefficients for h(z)=y(z)/x(z)
    returns b, a where b is numerator, a is denominator coefficients

    more or less a direct implementation of
    http://www.cs.princeton.edu/~ken/stmcb.pdf '''
    if len(x) != len(y):
        raise Exception("input and output vector sizes must match",
                        len(x), len(y))
    a = np.zeros(NP+1)
    a[0] = 1
    b = np.zeros(NZ+1)
    b[0] = 1
    X = scipy.linalg.toeplitz(x, np.zeros(NZ+1))
    Y = scipy.linalg.toeplitz(np.append([0], y[:-1]), np.zeros(NP))
    Q = np.hstack((X, Y))
    for i in range(N):
        # pre-filter x and y into xhat and yhat
        xhat = scipy.signal.lfilter([1], a, x)
        yhat = scipy.signal.lfilter([1], a, y)
        XH = scipy.linalg.toeplitz(xhat, np.zeros(NZ+1))
        YH = scipy.linalg.toeplitz(np.append([0], yhat[:-1]), np.zeros(NP))
        P = np.hstack((XH, YH))
        c = np.linalg.solve(np.dot(P.T, Q), np.dot(P.T, y))
        b = c[:NZ+1]
        a[1:] = -c[NZ+1:]
    return b, a
