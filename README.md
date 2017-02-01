This is a collection of hacks, experiments, and various detritus from my quest
to accurately emulate the TB-303 synthesizer.

A demo of what this code does is available here:
https://www.a1k0n.net/code/x0x/

(the source code for the above is also in this repo, currently index.html and
303.js)

I started with a few samples of 303 basslines found randomly on the Internet
and began trying to fit filter responses to them, by numerical analysis, by
fitting theoretical models, by trial and error.

One thing you can do is to take a single period of the input and do an FFT on
it to get the harmonic series, and compare against the spectrum of an ideal
sawtooth wave -- this will give you the filter magnitude and response curve.

I eventually got very frustrated with this approach and decided to buy an
actual TB-303 and poke at it with an oscilloscope... but I didn't want to spend
$2000 on this project, so I then decided to build my own x0xb0x, which was a
hugely valuable experience.

With an actual device to test (note: the x0xb0x uses the exact same circuit and
components as the original 303 for the analog section), I found that the diode
ladder filter of the 303 works almost exactly like the theoretical model says
it should, except for a high-pass filter in the resonance feedback loop.
Accounting for this is fairly easy mathematically and gives a very accurate
simulation of the filter.

One of the things that frustrated my earlier sound sample analysis approach is
that there are several other high-pass filtering effects between the 303 filter
output and the output amplifier, so the curves didn't match up. The 303
actually loses quite a lot of low-end bass sound between the filter and
amplifier as a result.

However accurately you simulate the filter, however, you won't get the right
sound unless you also simulate the filter cutoff envelope for all settings of
the "cutoff" and "env.mod" knobs.  Again, using an oscilloscope to measure the
current through the filter is hugely instructive, as the filter cutoff is
directly proportional to the current.

I now have a fairly decent filter simulation but a pretty crummy envelope
simulation and no accent simulation at all; I will update this build log when I
get any further.

Most of the code in here is garbage, but I did discover and implement a number
of neat things, like the Prony and Steiglitz-McBride algorithms to fit IIR
filters to arbitrary inputs and outputs...which I ultimately didn't use.
