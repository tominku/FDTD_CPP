import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path = "/home/minku/.data/output_cpu.txt"
#path = "/home/minku/.data/output_matlab.txt"
output = open(path, "r")
info_line = output.readline()

info = info_line.split(",")
Nx = int(info[0])
Ny = int(info[1])
N = Nx * Ny

material_image = np.zeros((Nx, Ny))
path = "/home/minku/.data/output_material.txt"
output = open(path, "r")
material_values = output.read().split(',')
#print(material_values)
material_values_len = len(material_values)
assert( N == material_values_len )
print(f'N: {N}, material_values_len: {material_values_len}')
for value, k in zip(material_values, range(len(material_values))):
    val = float(value)
    i = int(k % Nx)
    j = int(k / Nx)
    material_image[i, j] = val

#plt.imshow(material_image)
#plt.show()


path = "/home/minku/.data/output_cpu.txt"
#path = "/home/minku/.data/output_matlab.txt"
output = open(path, "r")
info_line = output.readline()
info = info_line.split(",")
Nx = int(info[0])
Ny = int(info[1])
images = []
steps = int(info[2])
logging_period = int(info[3])
print(f'Nx: {Nx}, Ny: {Ny}, Nt: {steps}, logging_period: {logging_period}')
min_value = 1e6
max_value = -1e6
frames = output.read().split(";")
#frames = frames[:50]
num_frames = len(frames)
print(f'num_frames: {num_frames}')
for frame, frame_i in zip(frames, range(num_frames)):
    image = np.zeros((Nx, Ny))
    #image = material_image.copy()
    frame_str_length = len(frame);
    if frame_str_length != 0:
        frame_values = frame.split(",");
        frame_values_len = len(frame_values)
        assert( N == frame_values_len )
        #print(f'frame_values_len: {frame_values_len}')
        min_value_in_frame = min_value
        max_value_in_frame = max_value
        for value, k in zip(frame_values, range(len(frame_values))):
            val = float(value)
            i = int(k % Nx)
            j = int(k / Nx)
            image[i, j] = val               
            if val < min_value_in_frame:
                min_value_in_frame = val
            elif val > max_value_in_frame:
                max_value_in_frame = val
                
        if frame_i > int(num_frames * 0.2):
            min_value = min_value_in_frame                
            max_value = max_value_in_frame
    else:
        print(f'no frame: {frame_str_length}');
        
    #image = image + material_image
    images.append(image)

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
    plt.title('%d / %d frame' % ((i * logging_period), steps))
    return [im]
#plt.colorbar(im)

interval_in_ms = 100
anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               interval = interval_in_ms, # in ms
                               frames=(num_frames),
                               blit=False                               
                               )
fps = int(1.0 / (interval_in_ms / 1000.0))
video_writer = animation.FFMpegWriter(fps=fps) 
anim.save('anim_Nx:%d_Ny:%d_logper:%d.mp4' % (Nx, Ny, logging_period), writer=video_writer)
#ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
#plt.colorbar(ax=im)
print('Done!')


# info = temp[0]
# info = info.split(",")
# Nx = int(info[0])
# Ny = int(info[1])
# steps = int(info[2])


# for step in range(steps):

#print(info)