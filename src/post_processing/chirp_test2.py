import numpy as np
import spectrum as sptr

chirp_duration_as_steps = 1000 # chirp duration as steps
dt = 0.01
T = chirp_duration_as_steps * dt
f0 = 1 # initial frequency
f1 = 10 # end frequency
k = (f1 - f0) / T # frequency change rate
N = 1000 + 1
ts = np.arange(chirp_duration_as_steps) # time points
ts = T * (ts / chirp_duration_as_steps)
print(ts)

#signal = np.cos(2*np.pi*(f0*ts))
signal = np.cos(2*np.pi*(f0*ts + (k/2)*np.power(ts, 2.0)) + np.pi/2)

#print(signal)

spt = sptr.Spectrum(signal, T)
freq, fft_result = spt.compute() 
spt.plot_spectrum(freq, fft_result)

import matplotlib.pyplot as plt

plt.plot(ts, signal)
plt.grid(True)
plt.show()