from __future__ import division  # normal division
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

T = 20
n = 512
t2 = np.linspace(-T / 2, T / 2, n + 1)
t = t2[1:n]

k = (2 * np.pi / T) * np.array(list(chain.from_iterable([range(0, int(n / 2 - 1), 1), range(int(-n / 2), 0, 1)])))

u = 2 / (np.exp(t) + np.exp(-t))  # hyperbolic secant
ud = -2 / (np.exp(t) + np.exp(-t)) * np.tanh(t)
u2d = 2 / (np.exp(t) + np.exp(-t)) - 2 * (2 / (np.exp(t) + np.exp(-t))) ** 3


ut = np.fft.fft(u).T
uds = np.fft.ifft((1j * k) * ut)
u2ds = np.fft.ifft((1j * k) ** 2 * ut)


plt.plot(t,ud,'r', t,uds,'mo')
plt.show()