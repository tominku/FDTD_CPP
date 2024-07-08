import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import json
import ujson as json
import time
import spectrum as sptr

begin = time.time()
path = "/home/minku/.data/EM_probe.json"
em_probes = []
with open(path, "r") as json_file:
    j = json.load(json_file)
    dt = j["dt"]
    probes = j["probes"]
    for probe in probes:
        name = probe["name"]
        print(f'name: {probe["name"]}')
        data = probe["data"]
        data = np.array(data, dtype=np.float32)
        print(data.shape)
        em_probe = {"name": name, "data": data}
        em_probes.append(em_probe)

# print(f"dt: {dt}")
# # plt.plot(em_probes[0]['data'], 'r-')
#plt.plot(em_probes[2]['data'][1500:2000])
# plt.plot(em_probes[1]['data'][1200:2000])
# #plt.plot(em_probes[2]['data'][:1500])
# #plt.plot(em_probes[2]['data'][:1500], 'o')
#plt.show()

# i_begin = 1200
# i_end = 2000
#i_begin = 1200
#i_end = 2000
i_begin = 1400
i_end = 2000

N = i_end - i_begin +1
T = dt * (N - 1)
signal = em_probes[2]['data'][i_begin:i_end]
spectrum = sptr.Spectrum(signal, T)
spectrum.print_info()
freq, fft_result = spectrum.compute()
bin_size = len(freq)
show_bin_size = int(bin_size / 3)
spectrum.plot_spectrum(freq[:show_bin_size], fft_result[:show_bin_size])
#spectrum.plot_spectrum(freq[:show_bin_size], 10*np.log10(fft_result[:show_bin_size]))