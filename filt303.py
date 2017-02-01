# theoretical model of a moog diode ladder filter with a one-pole high-pass
# filter in the feedback loop
import numpy as np


def laplace303(wf, fb, wc):
    ''' Laplace model of the 303 filter w/
    feedback cutoff frequency wf,
    feedback strength fb,
    cutoff frequency wc.
    
    The low-pass filter is a moog 4-pole diode ladder model w/ one of the
    capacitors halved (why did they do this? not sure), with feedback as 
    a simple high-pass RC filter.

    This is a fairly direct translation of the circuit into a mathematical
    model with minor simplifications.

    Returns numerator and denominator polynomials in the Laplace domain.
    '''

    N = np.poly1d([1.0/wf, 1])
    D = np.poly1d([1.0/wf, 1]) * np.poly1d([1.0/wc, 1])**3 \
        * np.poly1d([2.0/wc, 1]) + fb*(np.poly1d([1.0/wf, 0]))
    return N, D


''' One thing we can do with this is to make a giant look-up table for the
poles/zeros and transfer them to the z domain...

but most of the response is actually very linear, we could fairly easily do a
function approximation as well.

for now, let's do a case-by-case direct transform to the z domain by doing
pole-zero mapping and fixing the gain.
'''


def z303(T, wf, fb, wc):
    N, D = laplace303(wf, fb, wc)
    orig_gain = N(0) / D(0)
    z_zeros = np.exp(float(T) * np.roots(N))
    z_poles = np.exp(float(T) * np.roots(D))

    b = np.poly1d(z_zeros, True)
    a = np.poly1d(z_poles, True)
    gain = orig_gain * a(1) / b(1)
    b *= gain
    return b, a


#wc = 0.1
#RC = 2
#k = 3*(s*RC/(1+s*RC))**2
#for wc in [1.0, 2.0, 3.0, 4.0, 6.0]:
#        plt.loglog(-1j*s, np.abs(1 / ((1 + s * wc*0.1)**4 + k)), '-')
#
##s = 2j * np.exp(np.linspace(-4, np.log(100), 100))
