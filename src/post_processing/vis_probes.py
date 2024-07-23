import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import json
import ujson as json
import time
import spectrum as sptr

chirp_begin_freq = 6*1e9
chirp_end_freq = 10*1e9
dt = 0.00000000000707106769
fs = 1 / dt
total_steps = 2000
c = 299795648
T = total_steps * dt
S = (chirp_end_freq - chirp_begin_freq) / T

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


def get_peak_frequencies(freq, mags):
    detection_window_size = 3
    freq_len = freq.size
    last_index = freq_len - 1
    detected_frequencies = []
    for i in range(freq_len):
        if (i - detection_window_size) >= 0 and (i + detection_window_size) <= last_index:
            mag_window = mags[(i - detection_window_size):(i + detection_window_size)]
            mag_i = mags[i]
            local_max = np.max(mag_window)
            if mag_i >= local_max and local_max > 1.3*np.median(mag_window):         
                f = freq[i]       
                if f > chirp_end_freq:
                    continue
                else:
                    detected_frequencies.append(f)
                    #print(f'Detected Frequency {f / 1e9} GHz')

    return np.array(detected_frequencies)


# print(f"dt: {dt}")
#plt.plot(em_probes[0]['data'], label='Tx')
#plt.plot(em_probes[1]['data'], label='Rx')
#plt.legend()
#plt.plot(em_probes[2]['data'][1500:2000])
# plt.plot(em_probes[1]['data'][1200:2000])
# #plt.plot(em_probes[2]['data'][:1500])
# #plt.plot(em_probes[2]['data'][:1500], 'o')
#plt.show()

# i_begin = 1200
# i_end = 2000
#i_begin = 1200
#i_end = 2000
i_begin = 0
i_end = 2000 - 1

N = i_end - i_begin +1
T = dt * (N - 1)
signal_Tx = em_probes[0]['data'][i_begin:(i_end+1)]
signal_Rx = em_probes[1]['data'][i_begin:(i_end+1)]
print(f'Tx signal shape: {signal_Tx.shape}')
signal_mixed = signal_Tx * signal_Rx
#plt.plot(signal_mixed, label='Mixed')
print(f'mixed signal shape: {signal_mixed.shape}')
spectrum = sptr.Spectrum(signal_mixed, T)
spectrum_mixed = sptr.Spectrum(signal_mixed, T)
freq, fft_result = spectrum_mixed.compute()
peak_frequencies = get_peak_frequencies(freq, fft_result)
print(f'peak frequencies: {peak_frequencies / 1e9} GHz')
if peak_frequencies.size > 0:
    d = peak_frequencies[0] * c / (2*S)
    print(f'estimated d: {d} m')

#spectrum = sptr.Spectrum(signal_Tx, T)
#spectrum = sptr.Spectrum(signal_Rx, T)
spectrum.print_info()
freq, fft_result = spectrum.compute()
bin_size = len(freq)
show_bin_size = int(bin_size)
spectrum.plot_spectrum(freq[:show_bin_size], fft_result[:show_bin_size])
if peak_frequencies.size > 0:
    plt.plot(peak_frequencies/1e9, np.zeros_like(peak_frequencies), 'ro')
plt.show()
#spectrum.plot_spectrum(freq[:show_bin_size], 10*np.log10(fft_result[:show_bin_size]))

