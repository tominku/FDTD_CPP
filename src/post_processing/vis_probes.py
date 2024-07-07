import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import json
import ujson as json
import time

begin = time.time()
path = "/home/minku/.data/EM_probe.json"
em_probes = []
with open(path, "r") as json_file:
    probes = json.load(json_file)
    for probe in probes:
        name = probe["name"]
        print(f'name: {probe["name"]}')
        data = probe["data"]
        data = np.array(data, dtype=np.float32)
        print(data.shape)
        em_probe = {"name": name, "data": data}
        em_probes.append(em_probe)

# plt.plot(em_probes[0]['data'], 'r-')
plt.plot(em_probes[0]['data'][1200:2000])
plt.plot(em_probes[1]['data'][1200:2000])
#plt.plot(em_probes[2]['data'][:1500])
#plt.plot(em_probes[2]['data'][:1500], 'o')
plt.show()