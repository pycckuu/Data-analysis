## Data-analysis ##

=============
### dervatives_with_fft.py ###
Accuracy comparison between second- and fourth-order finite differ- ence methods and the spectral FFT method for calculating the first derivative. Note that by using the axis command, the exact solution (line) and its approx- imations can be magnified near an arbitrary point of interest. Here, the top figure shows that the second-order finite difference (circles) method is within O(10−2) of the exact derivative. The fourth-order finite difference (star) is within O(10−5) of the exact derivative. Finally, the FFT method is within O(10−6) of the exact derivative. This demonstrates the spectral accuracy prop- erty of the FFT algorithm

### filter.py ###
Time-domain (top) and frequency-domain (bottom) plots for a single realization of white-noise. In this case, the noise strength has been increased to ten, thus burying the desired signal field in both time and frequency. White-noise inundated signal field in the frequency domain along with a Gaussian filter with bandwidth parameter τ = 0.2 centered on the signal center frequency. (middle) The post-filtered signal field in the frequency domain. (bottom) The time-domain reconstruction of the signal field (bolded line) along with the ideal signal field (light line) and the detection threshold of the radar (dotted line).

### averaging.py ###
Shows the results from the averaging process for a nonstationary signal. The top graph shows that averaging over the time domain for a moving signal produces no discernible signal. However, averaging over the frequency domain produces a clear signature at the center-frequency of interest. Ideally, if more data is collected a better average signal is produced. However, in many applications, the acquisition of data is limited and decisions must be made upon what are essentially small sample sizes.

### windowed_fft.py ###
