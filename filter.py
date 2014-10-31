from __future__ import division  # normal division
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import random

T = 30
n = 512
t2 = np.linspace(-T / 2, T / 2, n + 1)
t = t2[1:n]

k = (2 * np.pi / T) * np.array(list(chain.from_iterable([range(0, int(n / 2 - 1), 1), range(int(-n / 2), 0, 1)])))  # rescaling to 2pi


u = 2 / (np.exp(t) + np.exp(-t))  # hyperbolic secant

ks = np.fft.fftshift(k)
ut = np.fft.fft(u).T

# white noise
r = np.random.randn(1, n - 1)
noise = 40
utn = ut + noise * (np.random.randn(1, n - 1)[0] + 1j * np.random.randn(1, n - 1)[0])
un = np.fft.ifft(utn)


filt = np.exp(-(k+15)**2)

utnf = filt*utn
unf = np.fft.ifft(utnf)

plt.subplot(2, 1, 1)
# plt.plot(t, u)
plt.plot(t,0*t+0.5)
# plt.plot(t,un,'m')
plt.plot(t,abs(unf),'g')
plt.xlim(-15, 15)
plt.subplot(2, 1, 2)
# plt.plot(ks, abs(np.fft.fftshift(ut)) / max(abs(np.fft.fftshift(ut))))
plt.plot(ks, abs(np.fft.fftshift(utn)) / max(abs(np.fft.fftshift(utn))),'m')
plt.plot(ks, abs(np.fft.fftshift(filt))/max(abs(np.fft.fftshift(filt))),'b')
plt.plot(ks, abs(np.fft.fftshift(utnf))/max(abs(np.fft.fftshift(utnf))),'g')
plt.xlim(-25, 25)


plt.show()
