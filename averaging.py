# Convergence to signal through repeated sampling

from __future__ import division  # normal division
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


T = 30  # time
n = 512  # points
t2 = np.linspace(-T / 2, T / 2, n + 1)  # [-15,15] 512 points
t = t2[1:n]

# wave numbers, frequency components rescaled to 2 pi: will be cos(0x), cos(1x), cos(2x), etc.. and shifted
k = (2 * np.pi / T) * np.array(list(chain.from_iterable([range(0, int(n / 2 - 1), 1), range(int(-n / 2), 0, 1)])))
ks = np.fft.fftshift(k)

u = 2 / (np.exp(t) + np.exp(-t))  # hyperbolic secant
ut = np.fft.fft(u).T

# white noise
# re-sampling of the signal and averaging / 30
noise = 60
ave = 0
sampling_count = 30  # how many times to re-sample the signal and average the noise
for i in xrange(1, sampling_count):
    utn = ut + noise * (np.random.randn(1, n - 1)[0] + 1j * np.random.randn(1, n - 1)[0])
    ave += utn

# utn/=sampling_count

ave = abs(np.fft.fftshift(ave)) / sampling_count

# Original signal in red
plt.plot(ks, abs(np.fft.fftshift(ut)), 'r', label='original signal')

# Signal with noise
plt.plot(ks, abs(np.fft.fftshift(utn)), 'c', label='signal with noise')

# re-sampled over and over again (30 times) and averaged signal with noise in black
plt.plot(ks, ave, 'b', label='averaged signal')
plt.legend(loc='upper right', shadow=True)


plt.show()
