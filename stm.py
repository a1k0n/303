import numpy as np
import scipy.signal
from matplotlib import pyplot as plt


def stm(y, x, NZ, NP, N, leadin=0):
    ''' steiglitz-mcbride N iterations, NZ zeros, NP poles
    input x(n), output y(n), returns filter coefficients for h(z)=y(z)/x(z)
    returns b, a where b is numerator, a is denominator coefficients.

    leadin is the number of x/y samples to ignore in the least-squares fitting
    procedure (but to use in the pre-filtering procedure), so that the filter
    can converge. If the input is not a raw impulse response but the middle of
    some waveform, then this is necessary.

    more or less a direct implementation of
    http://www.cs.princeton.edu/~ken/stmcb.pdf '''
    if len(x) != len(y):
        raise Exception("input and output vector sizes must match",
                        len(x), len(y))
    x_with_leadin = x
    x = x[leadin:]
    y_with_leadin = y
    y = y[leadin:]

    a = np.zeros(NP+1)
    a[0] = 1
    b = np.zeros(NZ+1)
    b[0] = 1
    X = scipy.linalg.toeplitz(x, np.zeros(NZ+1))
    Y = scipy.linalg.toeplitz(np.append([0], y[:-1]), np.zeros(NP))
    Q = np.hstack((X, Y))
    for i in range(N):
        # pre-filter x and y into xhat and yhat
        xhat = scipy.signal.lfilter([1], a, x_with_leadin)[leadin:]
        yhat = scipy.signal.lfilter([1], a, y_with_leadin)[leadin:]
        # plt.plot(xhat)
        # plt.plot(yhat)
        # plt.show()
        XH = scipy.linalg.toeplitz(xhat, np.zeros(NZ+1))
        YH = scipy.linalg.toeplitz(np.append([0], yhat[:-1]), np.zeros(NP))
        P = np.hstack((XH, YH))
        c = np.linalg.solve(np.dot(P.T, Q), np.dot(P.T, y))
        b = c[:NZ+1]
        a[1:] = -c[NZ+1:]
    return b, a
