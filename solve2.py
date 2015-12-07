import autograd.numpy as np
import autograd
import scipy.optimize


def FilterResponse(z, x):
    ''' fixed filter topology:
    w, pr12, pi12, p3, p4, z1 = x

    poles 1 and 2 are complex conjugates, and 3 is on the real axis; together
    they form a 3-pole lowpass filter.

    pole 4 and zero 1 are paired up opposite the imaginary axis and slightly
    offset to the -real direction; together they form an allpass + high pass
    filter to get closer to the real machine's response

    everything is scaled in the laplace plane by cutoff frequency w.

    we have a parameter degeneracy which is resolved by optimizing multiple
    time windows with the same topology, only changing w
    '''

    pr12, pi12, p3, p4, z1 = x[:5]
    w = x[5:]
    p1 = np.exp(w*(pr12 + 1j*pi12))
    p2 = np.conj(p1)
    p3 = np.exp(w*p3)
    p4 = np.exp(w*p4)
    z1 = np.exp(w*z1)
    z = z[:, np.newaxis]

    h = (z-z1) / (
        (z-p1) * (z-p2) * (z-p3) * (z-p4))
    return (h / h[0]).T


def solve(target, NH, T, x0):
    targetH = np.abs(target)

    def filterr(x):
        f = np.arange(1, NH)
        w = 2 * np.pi * f / T
        z = np.exp(1j*w)
        weight = np.log(f+1) - np.log(f)
        h = FilterResponse(z, x)
        h = np.abs(h)
        e = np.clip(h, 1e-2, 1e2) - np.clip(targetH[:, f], 1e-2, 1e2)
        err = weight * np.real(e * np.conj(e))
        return np.sum(err)

    xopt = scipy.optimize.fmin_ncg(
        filterr, x0, fprime=autograd.grad(filterr))
    return xopt
