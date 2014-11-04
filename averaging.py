from __future__ import division  # normal division
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import random

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
noise = 20
ave = 0
sampling_count = 30
for i in xrange(1,sampling_count):
    utn = ut + noise * (np.random.randn(1, n - 1)[0] + 1j * np.random.randn(1, n - 1)[0])
    un = np.fft.ifft(utn)
    ave +=utn
utn/=sampling_count

ave = abs(np.fft.fftshift(ave))/sampling_count
print ave

# plt.subplot(2, 1, 1)
# plt.plot(t, u, 'r')
# plt.plot(t, abs(un), 'k')
# plt.plot(t,0*t+0.5)
# plt.plot(t,un,'m')
# plt.plot(t,abs(unf),'g')
# plt.xlim(-15, 15)
# plt.subplot(2, 1, 2)
plt.plot(ks, abs(np.fft.fftshift(ut)), 'r')
plt.plot(ks, ave, 'k')

# plt.plot(ks, abs(np.fft.fftshift(utn)), 'k')
# plt.plot(ks, abs(np.fft.fftshift(utn)) / max(abs(np.fft.fftshift(utn))),'m')
# plt.plot(ks, abs(np.fft.fftshift(filt))/max(abs(np.fft.fftshift(filt))),'b')
# plt.plot(ks, abs(np.fft.fftshift(utnf))/max(abs(np.fft.fftshift(utnf))),'g')
# plt.xlim(-25, 25)


plt.show()
