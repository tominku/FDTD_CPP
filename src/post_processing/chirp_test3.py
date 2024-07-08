import numpy as np
import matplotlib.pyplot as plt

def chirp(t, f0, k, phi=0):
    """
    Generate a chirp signal.
    
    Parameters:
    t : array
        Time array
    f0 : float
        Initial frequency
    k : float
        Chirp rate
    phi : float, optional
        Initial phase (default is 0)
    
    Returns:
    array : Chirp signal
    """
    return np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + phi)

# Parameters
duration = 1.0  # seconds
fs = 1000  # sampling frequency
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

f0 = 10  # initial frequency
k = 100  # chirp rate

# Generate chirp signal
signal = chirp(t, f0, k)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title('Chirp Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot spectrogram
plt.figure(figsize=(10, 6))
plt.specgram(signal, Fs=fs, NFFT=256, noverlap=128)
plt.title('Spectrogram of Chirp Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity (dB)')
plt.show()