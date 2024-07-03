import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import json
import ujson as json
import time

begin = time.time()
path = "/home/minku/.data/material.json"
with open(path, "r") as json_file:
    material = json.load(json_file)
    size = material["material_data_size"]
    data = material["material_data"]
    Nx = material["Nx"]
    Ny = material["Ny"]
    N = Nx * Ny
    material_image_1D = np.array(data, dtype=np.float32)
    material_image_2D = np.reshape(material_image_1D, (Nx, Ny), order='F')

material_image = material_image_2D
#plt.imshow(material_image)
#plt.show()
end = time.time()
print(f'elapsed time loading material file {end - begin} seconds')


begin = time.time()
path = "/home/minku/.data/output_cpu.json"
with open(path, "r") as json_file:
    sim_data = json.load(json_file)    
    Nx = sim_data["Nx"]
    Ny = sim_data["Ny"]
    logging_period = sim_data["logging_period"]
    N = Nx * Ny    

t = 0
images = []
min_value = 1e6
max_value = -1e6
while(True):
    time_stamp = f't{t}'
    if not time_stamp in sim_data:
        break
    frame = sim_data[time_stamp]
    image_1D = np.array(frame, dtype=np.float32)
    if t > 100:
        max_temp = max(image_1D)
        min_temp = min(image_1D)
        if max_temp > max_value:
            max_value = max_temp
        if min_temp < min_value:
            min_value = min_temp
    image_2D = np.reshape(image_1D, (Nx, Ny), order='F')
    images.append(image_2D)
    t += logging_period

end = time.time()
print(f'elapsed time loading sim file {end - begin} seconds')

num_frames_to_show = 200

images_normalized = []
value_range = (max_value - min_value)
for image in images: # normalize images
    image = (image - min_value) / value_range
    image = (image * 2) - 1
    images_normalized.append(image)


print(f'min_value: {min_value}, max_value: {max_value}')

#fig = plt.figure( figsize=(12,12) )
fig = plt.figure( figsize=(Ny / 15, Nx / 15) )
#fig = plt.figure()

print(f'material: min {np.min(material_image)} max {np.max(material_image)}')

a = images_normalized[0]
cmap = "afmhot"
#im = plt.imshow(a, interpolation='none', cmap='gray', aspect='auto', vmin=0, vmax=1)
#im = plt.imshow(a, interpolation='none', cmap='gray', aspect='auto', vmin=min_value, vmax=max_value)
#im = plt.imshow(a, interpolation='none', cmap='viridis', aspect='auto', vmin=min_value, vmax=max_value, alpha=(1-material_image))
im = plt.imshow(a, interpolation='none', cmap=cmap, aspect='auto', vmin=-1, vmax=1, alpha=(1-material_image))
#plt.colorbar()

def animate_func(i):
    im.set_array(images_normalized[i])
    plt.title('%d / %d frame' % ((i * logging_period), 2000))
    return [im]
#plt.colorbar(im)

interval_in_ms = 100
anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               interval = interval_in_ms, # in ms
                               frames=(num_frames_to_show),
                               blit=False                               
                               )
fps = int(1.0 / (interval_in_ms / 1000.0))
video_writer = animation.FFMpegWriter(fps=fps) 
#anim.save('anim_Nx:%d_Ny:%d_logper:%d.mp4' % (Nx, Ny, logging_period), writer=video_writer)
#ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
#plt.colorbar(ax=im)
print('Done!')