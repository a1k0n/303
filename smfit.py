#!/usr/bin/env python
''' Steiglitz-McBride fitting of 303 filter.

I've learned quite a few new tricks since the last time I attempted this:

 - Use cepstrum method to get the minimum-phase impulse response from an
   arbitrary frequency magnitude
 - Weight the minimum phase impulse response by frequency appropriately (here,
   it's sufficient to simply zero out DC and leave everything else alone) and
   do the same for the unit impulse waveform
 - Use the Steiglitz-McBride algorithm to fit the minimum phase filter between
   the preweighted impulse to the preweighted minimum phase response.
 - Do the above procedure for each full waveform (we don't need to even try to
   line up with the sawtooth waveform on the input!) and do a linear regression
   on each pole and zero throughout the note.

The 303 filter fits without audible differences with two zeros and four poles.

Everything is least-squares, no iteration necessary.
'''

import wave
import numpy as np
import stm

T = 5099  # sequencer step speed on saw03.wav


def Load():
    w = wave.open("saw03.wav")
    wav = np.frombuffer(w.readframes(w.getnframes()), np.int16)[::2]
    w.close()
    notes = []
    # grab the 5th note in the pattern, which runs for 3 steps (the longest)
    for i in range(0, len(wav)/(32*T)):
        notes.extend(wav[i*32*T + 6*T:i*32*T + 9*T])
    return notes, wav


def minphase(x, y):
    ''' Return the minimum-phase impulse response and frequency response given
    an input and output sequence.
    N.B.: not totally general as it discards the DC component from both. '''
    # compute the cepstrum and truncate its negative frequency components
    # to get a minimum-phase impulse response for h=y/x
    Y = np.fft.fft(y)
    Y[0] = 1
    X = np.fft.fft(x, len(Y))
    X *= np.abs(Y[1]) / np.abs(X[1])
    X[0] = 1
    h = Y / X
    hmin = np.fft.ifft(np.log(h))
    # DC 1 2 3 4 5 ... -5 -4 -3 -2 -1
    nt = (len(h) - 1) / 2
    # fold the negative frequency part of the cepstrum ifft back onto the
    # positive frequencies
    hmin[1:nt+1] += np.conj(hmin[:-nt-1:-1])
    hmin[-nt:] = 0
    # now undo the ifft, the exp, and the fft to get an impulse response back
    hh = np.exp(np.fft.fft(hmin))
    hh[0] = 0  # and remove DC, as we don't want to try to fit it later
    return np.real(np.fft.ifft(hh)), hh


def fit303saw(o, T, T2, N, NZ, NP):
    ''' Find the best matching filter sweep to the input 303 sawtooth waveform
    o is the output to match, T is the fundamental period, T2 is the step size
    and N is the number of steps to generate filters for. NZ, NP control the
    number of zeros and poles in the filter fit. '''

    zeros = np.zeros((N, NZ), np.complex128)
    poles = np.zeros((N, NP), np.complex128)
    T = int(T)
    d = np.zeros(T)  # unit impulse (dirac delta)
    d[0] = 1
    d -= np.mean(d)  # ...except with mean removed
    saw = np.linspace(-1, 1, T)
    for i in range(N):
        y = notes[int(o):int(o)+T]
        o += T2
        imp, h = minphase(saw, y)
        imp -= np.mean(imp)  # should already be zeroed out
        b, a = stm.stm(imp, d, NZ, NP, 20)
        zeros[i] = np.log(np.roots(b))
        poles[i] = np.log(np.roots(a))
    return zeros, poles


def fitroots(poles):
    ''' return a linear fit to the roots (poles or zeros or whatever) '''
    X = np.vstack((np.arange(len(poles)), np.ones(len(poles)))).T
    linp = np.linalg.lstsq(X, poles)
    print linp
    return linp[0], np.dot(X, linp[0])


if __name__ == '__main__':
    notes = Load()
    for o in range(800, len(notes), 3*T):
        zeros, poles = fit303saw(o, 611.9, 611.9, 20, 2, 4)
        zfit, _ = fitroots(zeros)
        pfit, _ = fitroots(poles)
        print o, 'zfit', zfit
        print o, 'pfit', pfit
