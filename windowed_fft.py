# Moving signals and averaging of shifted frequency
# Signal detection through averaging and averaging
# windowed FFT

from __future__ import division  # normal division
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T = 60  # time
n = 512  # points
t2 = np.linspace(-T / 2, T / 2, n + 1)  # [-15,15] 512 points
t = t2[1:n]

# wave numbers, frequency components rescaled to 2 pi: will be cos(0x), cos(1x), cos(2x), etc.. and shifted
k = (2 * np.pi / T) * np.array(list(chain.from_iterable([range(0, int(n / 2 - 1), 1), range(int(-n / 2), 0, 1)])))
ks = np.fft.fftshift(k)


slice = np.arange(0, 10, 0.5)
[T, S] = np.meshgrid(t, slice)
[K, S] = np.meshgrid(k, slice)


# moving signal in time and shifted frequency by 10(for instance due to Doppler effect). Means everything multiplied by cos(10x)
U = 2 / (np.exp(T - 10 * np.sin(S)) + np.exp(-(T - 10 * np.sin(S)))) * np.exp(1j * 10 * T)
# get rid of phase due to shifted frequency.
# U = abs(U)

# adding random noise and fft of moving signal
noise = 20

UT = abs(np.fft.fftshift(np.fft.fft(U + noise * (np.random.standard_normal((n-1,len(slice))).T + 1j * np.random.standard_normal((n-1,len(slice))).T))))
UN = np.fft.fftshift(UT)

# also, I don't know what frequency I am looking for. There, I need windowed FFT:

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_surface(T, S, UN)
ax.set_xlabel('domain')
ax.set_ylabel('time')
ax.set_zlabel('signal')
ax.view_init(elev=40., azim=300)


ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_surface(np.fft.fftshift(K), S, UT)
ax2.set_xlabel('domain')
ax2.set_ylabel('time')
ax2.set_zlabel('signal')
ax2.view_init(elev=40., azim=300)


plt.show()

# for i in xrange(1, len(slice)):
    # UT[i,:] = abs(np.fft.fftshift(np.fft.fft(U[i,:])))



# u = 2 / (np.exp(t) + np.exp(-t))  # hyperbolic secant
# ut = np.fft.fft(u).T

# white noise
# re-sampling of the signal and averaging / 30
# noise = 20
# ave = 0
# sampling_count = 30  # how many times to re-sample the noise and average the noise
# for i in xrange(1, sampling_count):
#     utn = ut + noise * (np.random.randn(1, n - 1)[0] + 1j * np.random.randn(1, n - 1)[0])
#     ave += utn

# utn/=sampling_count

# ave = abs(np.fft.fftshift(ave)) / sampling_count

# Original signal in red
# plt.plot(ks, abs(np.fft.fftshift(ut)), 'r', label='original signal')

# Signal with noise
# plt.plot(ks, abs(np.fft.fftshift(utn)), 'c', label='signal with noise')

# re-sampled over and over again (30 times) and averaged signal with noise in black
# plt.plot(ks, ave, 'b', label='averaged signal')
# plt.legend(loc='upper right', shadow=True)
