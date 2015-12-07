""" Take an audio sample, attempt to reconstruct the filter used to generate it
"""

import wave
import autograd.numpy as np
import autograd
from autograd.util import quick_grad_check
import scipy.optimize
import matplotlib.pyplot as plt


def LoadWAV(fname):
    w = wave.open(fname)
    x = w.readframes(w.getnframes())
    return np.frombuffer(x, np.int16)


def Fundamental(Y):
    # initial guess: use peak of FFT to get to within one sample period
    T = float(len(Y)) / np.argmax(np.abs(np.fft.fft(Y)[:2000]))

    # (not really necessary, but why not) refine
    def mag(T):
        w = np.arange(0, len(Y)) * 2 * np.pi / T
        x = Y * np.exp(1j * w)  # dx/dw = j Y exp(jw)
        # dx*/dw = -j Y exp(-jw)
        # d/dw = x* dx/dw + x dx*/dw
        #      = Y exp(1j*w)
        return np.abs(np.sum(x))

    # print T, mag(T)
    # print T+0.1, mag(T+0.1)
    # print T-0.1, mag(T-0.1)
    # print round(T), mag(round(T))

    return round(T)


def Envelope(Y, T):
    N = len(Y) // T  # number of envelope points; one for each period of wave
    # get magnitude of fundamental only
    mag = Y[:N*T] * np.exp(2j * np.pi * np.arange(N*T) / T)
    return np.abs(np.sum(mag.reshape((N, T)), axis=1)) / T


Y = LoadWAV("TB-303 Bass 01.wav")
#Y = LoadWAV("TB-303 Bass 18.wav")
T = Fundamental(Y)
saw0 = np.linspace(-0.5, 0.5, T)
X = np.fft.fft(saw0)
X /= X[1]
y, targetH = None, None


def LoadPeriod(n):
    global y, targetH
    y = np.fft.fft(Y[int(n*T):int((n+1)*T)])
    y /= np.abs(y[1])
    jw = np.log(y[1])
    phase = np.exp(-jw*np.arange(len(y)))
    targetH = y * phase / X


def FilterSPolesZeros(x):
    # return 4 poles and 1 zero
    p1i = x[0] < 0 and -x[0]*1j or x[0]
    p2i = x[2] < 0 and -x[2]*1j or x[2]
    return ([x[1] + p1i, x[1] - p1i,
             x[3] + p2i, x[3] - p2i],
            [x[4]])

 
# define target filter here and in FilterCoeffs
def FilterResponse(z, x):
    ''' -b +- sqrt(b^2 - 4c) / 2
    so we want to independently control b and the discriminant
    which really just turns out to be + -> real, - -> imag
    so we allow the "frequency" component to go positive or negative
    if it's negative we use complex conjugates
    if it's positive we use "real conjugates"
    s^2 + bs + c = (s-p1)(s-p2)
    p1,p2 = b/2 +- sqrt(b^2 - 4ac)/2 = x1 +- sqrt(x2))
    b = x1*2
    sqrt(b^2 - 4c)/2 = sqrt(x2)
    b^2/4 - c = x2
    c = b^2/4 - x2
    wait, the quadratic form isn't all that useful though; we need to map the
    poles and zeros so we might as well just make it a distance either in real
    or in imag
    '''
    p1i = x[0] < 0 and -x[0]*1j or x[0]
    p1 = np.exp(p1i + x[1])
    p1c = np.exp(-p1i + x[1])
    p2i = x[2] < 0 and -x[2]*1j or x[2]
    p2 = np.exp(p2i + x[3])
    p2c = np.exp(-p2i + x[3])
    z1 = np.exp(x[4])
    if np.imag(p1) == 0 and np.real(p1) > 1.0:
        p1 = 1.0 - p1
    if np.imag(p1c) == 0 and np.real(p1c) > 1.0:
        p1c = 1.0 - p1c
    if np.imag(p2) == 0 and np.real(p2) > 1.0:
        p2 = 1.0 - p2
    if np.imag(p2c) == 0 and np.real(p2c) > 1.0:
        p2c = 1.0 - p2c
    h = (z - z1) / (
        (z-p1) * (z-p1c) * (z-p2) * (z-p2c))
    return h


def FilterCoeffs(x):
    """ return filter coefficients """
    p1i = x[0] < 0 and -x[0]*1j or x[0]
    p1 = np.exp(p1i + x[1])
    p1c = np.exp(-p1i + x[1])
    p2i = x[2] < 0 and -x[2]*1j or x[2]
    p2 = np.exp(p2i + x[3])
    p2c = np.exp(-p2i + x[3])

    if np.imag(p1) == 0 and np.real(p1) > 1.0:
        p1 = 1.0 - p1
    if np.imag(p1c) == 0 and np.real(p1c) > 1.0:
        p1c = 1.0 - p1c
    if np.imag(p2) == 0 and np.real(p2) > 1.0:
        p2 = 1.0 - p2
    if np.imag(p2c) == 0 and np.real(p2c) > 1.0:
        p2c = 1.0 - p2c
    z1 = np.exp(x[4])
    # first biquad:
    # (z - z1) / [(z - p1) (z - p1*)]
    a1 = np.real(p1 + p1c)
    b1 = -np.real(p1 * p1c)

    # second:
    # 1.0 / [(z - p2) (z - p3)]
    # (1 - p2 z^-1) (1 - p3 z^-1)
    a2 = np.real(p2 + p2c)
    b2 = -np.real(p2 * p2c)

    z = np.exp(2j * np.pi / T)
    gain = 1.0 / np.abs(FilterResponse(z, x))
    return np.array([gain, -z1, a1, b1, a2, b2])


def FilterInitialState():
    # x1, y11, y12, y21, y22
    return np.zeros(5)


# Y/X = (z - z1); Y = (1 - z1*z^-1)X
def FilterUpdate(coef, state, x):
    #       x1, y11, y12, y21, y22 = state
    # gain, z1, a1,  b1,  a2,  b2 = coef
    x *= coef[0]
    y = x + np.dot(state[0:3], coef[1:4])
    state[0] = x
    state[2] = state[1]
    state[1] = y
    y = y + np.dot(state[3:5], coef[4:6])
    state[4] = state[3]
    state[3] = y
    return y


def filterr(x):
    f = np.arange(1, 64)
    w = 2 * np.pi * f / 670.0
    z = np.exp(1j*w)
    weight = np.log(f+1) - np.log(f)
    # this is also a dc-blocking filter
    h = FilterResponse(z, x)
    h = h / h[0]
    #e = np.log(np.clip(h, 1e-2, 1e2)) - np.log(np.clip(targetH[f], 1e-2, 1e2))
    e = np.clip(h, 1e-2, 1e2) - np.clip(targetH[f], 1e-2, 1e2)
    err = weight * np.real(e * np.conj(e))
    # err = weight * (np.abs(h) - np.abs(targetH[f]))**2
    return np.sum(err)


def plotfilt(x):
    f = np.arange(1, 256)
    w = 2 * np.pi * f / 670.0
    z = np.exp(1j*w)
    h = FilterResponse(z, x)
    h = h / h[0]
    plt.plot(np.log(f) / np.log(2), 10*np.log(np.clip(np.abs(h), 1e-2, 1e2)))


def plotfiltphase(x):
    f = np.arange(1, 256)
    w = 2 * np.pi * f / 670.0
    z = np.exp(1j*w)
    h = FilterResponse(z, x)
    h = h * (h[1] / np.abs(h[1]))**(np.arange(len(h)) - 2)
    phase = np.angle(h[1:] / h[:-1])
    plt.plot(np.log(f[:-1]) / np.log(2), phase)
    # plt.plot(np.angle(h))


def plot303(n):
    f = np.arange(1, 300)
    y = np.abs(np.fft.rfft(Y[int(n*670):int((n+1)*670)]))
    ypeak = np.argmax(y)
    print "peak @ f=", np.log(ypeak) / np.log(2)
    plt.plot(np.log(f) / np.log(2), 10*np.log(
        np.clip(np.abs(y[f] / (y[1] * X[f])), 1e-2, 1e2)))


def plot303phase(n):
    f = np.arange(1, 300)
    y = np.fft.rfft(-Y[int(n*670)-100:int((n+1)*670)-100])
    # y = y * (y[1] / np.abs(y[1]))**(np.arange(len(y)) - 2)
    phase = np.angle(y[1:] / y[:-1])
    plt.plot(np.log(f) / np.log(2), phase[f-1])


vg = autograd.value_and_grad(filterr)
# x0 = np.array([0.17608125,  3.32796736,  1.2262284 ,  4.32030568,  5.18875782])
x0 = np.array([-0.17, -np.exp(-3), 0.1, -np.exp(-2), -np.exp(-5)])


def solve(n, x0):
    LoadPeriod(n)
    xopt = scipy.optimize.fmin_ncg(
        filterr, x0, fprime=autograd.grad(filterr))
    print n, xopt
    return xopt


def solveall(x0):
    n = int(len(Y) / T)
    x = x0
    xs = []
    for i in range(n):
        x = solve(i, x)
        xs.append(x)
        print x
    return np.array(xs)


def reconstruct():
    ''' Fit an exponential function to the pole locations over all periods in
    the source wave '''
    n = int(len(Y) / T)
    targetH = np.zeros((n, 63))
    saw = np.linspace(-0.5, 0.5, T)
    X = np.fft.fft(saw)
    X /= X[1]
    X = np.clip(np.abs(X[1:64]), 1e-2, 1e2)
    for i in range(0, n):
        y = np.abs(np.fft.fft(Y[int(i*T):int((i+1)*T)]))
        targetH[i, :] = np.log(np.clip(y[1:64] / (y[1] * X), 1e-2, 1e2))
    # plt.plot(targetH[10, :])

    f = np.arange(1, 64)
    weight = np.log(f+1) - np.log(f)
    w = 2 * np.pi * f / T
    z = np.exp(1j*w)
    x0 = wave01_params
    x0[0] = -4.5

    def err(x):
        err = 0
        for i in range(n):
            s = np.exp(-np.exp(x[0]) * i)
            y = [x[1] + x[2]*s, x[3] + x[4]*s,
                 x[5] + x[6]*s, x[7] + x[8]*s,
                 x[9] + x[10]*s]
            h = FilterResponse(z, y)
            h = h / h[0]
            err += np.mean(
                weight * (np.log(np.clip(np.abs(h), 1e-2, 1e2)) - targetH[i, :])**2)
        return err

    # quick_grad_check(err, x0)

    '''
    # print autograd.value_and_grad(err)(x)
    vg = autograd.value_and_grad(err)
    # solve w/ nesterov's accelerated gradient
    learn = 0.0001
    momentum = 0.9
    velocity = np.zeros(len(x))
    for i in range(200):
        v, g = vg(x + momentum * velocity)
        gg = np.dot(g, g)
        print v, gg, x
        velocity = momentum * velocity - learn*g
        x += velocity
        if gg < 1e-1:
            break
    return x
    '''
    def printcb(xk):
        print err(xk), xk

    xopt = scipy.optimize.fmin_ncg(
        err, x0, fprime=autograd.grad(err),
        maxiter=30, callback=printcb)
    return xopt


# fit exponential curve to data
def fitexp(y, forceT=None):
    t = np.arange(len(y))

    def err(x):
        T = x[0]
        if forceT is not None:
            T = forceT
        yy = x[1] + x[2] * np.exp(-t * np.exp(T))
        return np.mean((yy - y)**2)
    x0 = np.array([-3, y[-1], y[0] - y[-1]])
    # print quick_grad_check(err, x0)
    xopt = scipy.optimize.fmin_ncg(
        err, x0, fprime=autograd.grad(err))
    return xopt


'''
18.wav parameters:
decay rate: -t*exp(-3)
imag base, add, real base, add
pole pair 1:
-0.03933817, -0.14012724, -0.0102258  -0.02808014
pole pair 2:
0.03624802,  0.16378753, -0.04509206 -0.10511563
zero:
-0.00245105294529, -0.00430300831021
'''
wave18_params = np.array([
    -3.0,  # decay rate, exp(-t*exp(3))
    # imag base, add,         real base,   add
    -0.04049476, -0.13301186, -0.01022580, -0.02808014,
    0.03506089,  0.0985560,  -0.04509206, -0.10511563,
    -0.00245105294529, -0.00430300831021])

wave01_params = np.array(
    [-4.50441499e+00,  -1.21142437e-01,  -9.05807813e-02,
     -5.50964061e-03,  -1.51117713e-03,   7.54491803e-02,
     1.02082227e-01,  -8.93292836e-02,  -9.67523726e-02,
     5.21468801e-03,  -3.97596605e-03])


def getparams(x):
    n = int(len(Y) / T)
    for i in range(n):
        s = np.exp(-np.exp(x[0]) * i)
        yield [x[1] + x[2]*s, x[3] + x[4]*s,
               x[5] + x[6]*s, x[7] + x[8]*s,
               x[9] + x[10]*s]


def synth(T, params):
    saw = np.linspace(0, 1, T)
    saw[T//2:] -= 1.0
    params = list(params)
    out = np.zeros(T*len(params))
    """
    Y = X (1+z^-1) / (1 - 2 Re[p1] z^-1 + p1p1* z^-2)
    Y' = Y / (z^2 - 2 Re[p2] z^-1 + p2p2* z^-2)
    gain1 = 2 / (1 - 2 Re[p1] + p1p1*)
    gain2 = 1 / (1 - 2 Re[p2] + p2p2*)
    """
    state = FilterInitialState()
    n = 0
    for p in params:
        coef = FilterCoeffs(p)
        k = n*T
        env = np.exp(-0.007*n)
        for i in range(T):
            out[k+i] = env * FilterUpdate(coef, state, saw[i])
        n += 1
    return out
