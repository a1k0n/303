import autograd.numpy as np
import autograd
# import scipy.optimize
import matplotlib.pyplot as plt


def adam(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)  # , i)
        if callback:
            callback(x, i, g)
        m = (1 - b1) * g + b1 * m       # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return x


Pattern03 = {
    'dur':    [2, 2, 1, 1, 3, 1, 1, 2, 1, 2,
               2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1],
    'accent': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
               0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'periods': [306, 612, 308, 257, 612, 306, 308, 515, 515, 515,
                306, 612, 308, 257, 612, 306, 308, 515, 515, 515, 308]
}

'''
Pattern03:
                ^^^                     ^^^                 ^^^
D#1.... D#0.... D#1 F#1 D#0........ D#1 D#1 F#0.... F#0 F#0....
D#1.... D#0.... D#1 F#1 D#0.... D#1.... D#1.... F#0 F#0 F#0 D#1


The 303 controls are:
 - cutoff frequency (base of wc sweep)
 - resonance (pi12 = k / reso)
 - envelope modulation (exponential damped wc sweep added to cutoff)
 - decay (controls volume + cutoff falloff exponent)
 - accent (if accent is enabled on note, boosts envmod + initial volume +
   decay?)
   - according to http://www.firstpr.com.au/rwi/dfish/303-unique.html
     accent turns decay all the way down for the filter envelope generator.

TODO:
 - decode pattern notes / accents / slides
 - reparameterize note solver in terms of cutoff, envmod, decay, resonance
 - analyze accents
'''


def FilterResponse(z, x, n):
    ''' fixed filter topology:
    w, pr12, pi12, p3, p4, z1 = x

    poles 1 and 2 are complex conjugates, and 3 is on the real axis; together
    they form a 3-pole lowpass filter.

    pole 4 and zero 1 are paired up opposite the imaginary axis and slightly
    offset to the -real direction; together they form an allpass + high pass
    filter to get closer to the real machine's response

    everything is scaled in the laplace plane by cutoff frequency w.

    we have a parameter degeneracy w.r.t. w, so we'll fix pi12 to 1.0.
    '''

    Q, p3, p4, z1, w0, w1, wdecay = x
    w = w1 * np.exp(-np.arange(n) * wdecay) + w0
    p1 = np.exp(w*(-1.0 / Q + 1j))
    p2 = np.conj(p1)

    p3 = np.exp(w * p3 / Q)
    p4 = np.exp(w * p4 / Q)
    # p3 = np.exp(w * (p3 / Q + p4 * 1j))
    # p4 = np.conj(p3)

    # not sure whether p4/z1 should scale with cutoff.
    z1 = np.exp(w * z1)

    z = z[:, np.newaxis]

    h = (z-z1) * (z-z1) / (
        (z-p1) * (z-p2) * (z-p3) * (z-p4))
    return (h / h[0]).T


def FilterCoeffs(x, n):
    Q, p3, p4, z1, w0, w1, wdecay = x
    w = w1 * np.exp(-np.arange(n) * wdecay) + w0
    p1 = np.exp(w*(-1.0 / Q + 1j))
    p2 = np.conj(p1)
    p3 = np.exp(w * p3 / Q)
    p4 = np.exp(w * p4)
    z1 = np.exp(w*z1)

    gain = np.ones(n)  # FIXME

    return np.vstack((
        gain,
        -z1, np.real(p1 + p2), -np.real(p1 * p2),
        -z1, np.real(p3 + p4), -np.real(p3 * p4))).T


def PlotPoles(x, n):
    Q, p3, p4, z1, w0, w1, wdecay = x
    w = w1 * np.exp(-np.arange(n) * wdecay) + w0
    p1 = (w*(-1.0 / Q + 1j))
    p2 = np.conj(p1)
    p3 = (w * p3 / Q)
    p4 = (w * p4)
    z1 = (w * z1)

    a = np.vstack((p1, p2, p3, p4, z1))
    plt.axvline(0)
    plt.plot(np.real(a), np.imag(a), 'x')


def solve(target, NH, T, x0):
    f = np.arange(1, NH)
    w = 2 * np.pi * f / T
    z = np.exp(1j*w)
    weight = np.log(f+1) - np.log(f)
    # targetH = np.clip(np.log(np.abs(target[:, f])), 1e-2, 1e2)
    targetH = np.log(np.abs(target[:, f]))
    N = targetH.shape[0]

    def filterr(x):
        h = FilterResponse(z, x, N)
        h = np.log(np.abs(h))
        # e = np.clip(h, 1e-2, 1e2) - targetH
        e = h - targetH
        err = weight * np.real(e * np.conj(e))
        return np.sum(err)

    #xopt = scipy.optimize.fmin_ncg(
    #    filterr, x0, fprime=autograd.grad(filterr))
    def cb(x, i, g):
        if (i % 100) == 0:
            print i, filterr(x)
    g = autograd.grad(filterr)
    def grad(x, i):
        return g(x)
    xopt = adam(grad, x0, cb, 10000)
    return xopt


def SolveNote(sample, Tp):
    NH = 60
    H = []
    saw = np.fft.rfft(np.linspace(-0.5, 0.5, Tp))
    saw = saw/saw[1]

    # Identify volume envelope of note
    NP = len(sample) // Tp  # num periods
    print 'NP =', NP
    IQ = sample[:NP * Tp] * np.exp(np.pi * 2j * np.arange(NP * Tp) / Tp)
    vol = np.abs(IQ.reshape((NP, Tp)).sum(axis=1))
    plt.plot(vol, '-x')
    plt.axhline(0.3 * np.max(vol))
    plt.show()

    # Choose all periods with at least some % of the max magnitude
    thresh = 0.5 * np.max(vol)
    for i in range(NP):
        if vol[i] < thresh:
            continue
        f = np.fft.rfft(sample[i*Tp:i*Tp+Tp])
        f = f / (saw * f[1])
        f[0] = 0
        H.append(f)
    H = np.array(H)
    print 'NH =', H.shape
    HH = np.abs(H)

    # Guess some initial parameters
    c1 = np.argmax(HH[0])
    c0 = np.argmax(HH[-1])
    Q = HH[0, c1] / HH[0, 1]
    wc1 = 2 * np.pi * c1 / Tp
    wc0 = 2 * np.pi * c0 / Tp

    x0 = np.array([
        Q,  # resonance
        -6.0,  # 3rd pole location
        -0.3,  # 4th pole
        0.2,  # hi-pass
        wc0,  # cutoff
        wc1-wc0,  # envmod
        0.1    # decay (e^-kn where n is in wave periods)
    ])

    return H, solve(H, NH, Tp, x0)


def GetNotePeriod(x):
    F1 = np.abs(np.fft.fft(x))
    F2 = np.abs(np.fft.rfft(F1))
    # Find the peak inter-harmonic spacing
    # (but this won't work for square waves!)
    peak = np.argmax(F2[30:]) + 30
    return peak


def GetTempoPeriod(sample):
    F = np.fft.rfft(sample[:3488000])
    f = np.argmax(np.abs(F[:1000]))
    Tq = float(3488000) / f
    return Tq


def GetPeriods(sample, pattern, Tq):
    off = 0
    periods = []
    for d in pattern['dur']:
        o1 = off + Tq*d
        periods.append(GetNotePeriod(sample[int(off):int(o1)]))
        off = o1
    return periods


def ClosestNote(period):
    n = 12 * np.log(48000.0 / (period * 440.0)) / np.log(2) + 57
    n = int(round(n))
    return n
    #octave = 3 + n // 12
    #n %= 12
    #notename = 'A A#B C C#D D#E F F#G G#'[2*n:2*n+2]
    #return notename + str(octave)


def Resynth(fitcoef, Tp, N):
    out = []
    coef = FilterCoeffs(fitcoef, N)
    x = np.linspace(-0.5, 0.5, Tp)
    fstate1 = np.zeros(3)
    fstate2 = np.zeros(3)
    for i in range(N):
        c = coef[i]
        for j in range(Tp):
            y1 = x[j] * c[0] + np.dot(fstate1, c[1:4])
            fstate1[2] = fstate1[1]
            fstate1[1] = y1
            fstate1[0] = x[j] * c[0]

            y2 = y1 + np.dot(fstate2, c[4:7])
            fstate2[2] = fstate2[1]
            fstate2[1] = y2
            fstate2[0] = y1

            out.append(y2)

    return np.array(out)
