import spectrum as sptr
import numpy as np

T = 1e-9    # Total simulation time (s)
N = 100
# Create time array
t = np.linspace(0, T, N)
f1, f2 = 2e9, 5e9  # Frequencies of the sine waves
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

spectrum = sptr.Spectrum(signal, T)
spectrum.print_info()
freq, fft_result = spectrum.compute()
spectrum.plot_spectrum(freq, fft_result)