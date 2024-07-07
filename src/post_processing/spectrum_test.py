import spectrum as sptr
import numpy as np

T = 1e-9    # signal duration (s)
N = 100
# Create time array
t = np.arange(1, N+1)
delta_t = T / N
t = delta_t * t
f1, f2 = 2e9, 5e9  # Frequencies of the sine waves
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

spectrum = sptr.Spectrum(signal, T)
spectrum.print_info()
freq, fft_result = spectrum.compute()
spectrum.plot_spectrum(freq, fft_result)