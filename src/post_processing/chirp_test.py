import numpy as np

T = 5 # chirp duration
f0 = 1 # initial frequency
f1 = 10 # end frequency
k = (f1 - f0) / T # frequency change rate
N = 1000 + 1
ts = np.arange(N) # time points
ts = T * (ts / (N-1))
print(ts)

#signal = np.cos(2*np.pi*(f0*ts))
signal = np.cos(2*np.pi*(f0*ts + (k/2)*np.power(ts, 2.0)) + np.pi/2)

#print(signal)

import matplotlib.pyplot as plt

plt.plot(signal)
plt.grid(True)
plt.show()