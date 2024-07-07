import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

class Spectrum:
    
    def __init__(self, data: np.ndarray, T: int):
        self.N = len(data)
        self.data = data
        self.dt = T / (self.N - 1)
        #self.dt = T / (self.N)
        self.fs = 1.0 / self.dt
        self.delta_f = self.fs / self.N
    
    def print_info(self):
        print(f'N: {self.N}, dt: {self.dt}, fs: {self.fs / 1e9} GHz, delta_f: {self.delta_f / 1e9} GHz')

    def compute(self):
        fft_result = fft(self.data)
        fft_result = fft_result[:(self.N//2 + 1)]
        fft_result = np.abs(fft_result)
        freq = np.arange((self.N//2 + 1))
        freq = freq * self.delta_f

        return freq, fft_result
    
    def peak_frequency(self, freq, fft_result):
        pass
        # # Find and print the dominant frequencies
        # threshold = 0.1 * np.max(fft_result)  # 10% of the maximum amplitude
        # peaks = freq[fft_result > threshold] / 1e9
        # print(f"Dominant frequencies: {peaks} GHz")

    def plot_spectrum(self, freq, fft_result):
        plt.figure(figsize=(10, 6))
        plt.plot(freq / 1e9, fft_result)
        plt.plot(freq / 1e9, fft_result, 'ro')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum')
        plt.grid(True)
        plt.show()
