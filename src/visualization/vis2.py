import numpy as np

output = open("/home/minku/.data/output.txt", "r")
info_line = output.readline()

info = info_line.split(",")
Nx = int(info[0])
Ny = int(info[1])
images = []
steps = int(info[2])
N = Nx * Ny
print(f'Nx: {Nx}, Ny: {Ny}, Nt: {steps}')
min_value = 1e6
max_value = -1e6
frames = output.read().split(";")
for frame in frames:
    image = np.zeros((Nx, Ny))
    images.append(image)
    frame_str_length = len(frame);
    if frame_str_length != 0:
        frame_values = frame.split(",");
        frame_values_len = len(frame_values)
        assert( N == frame_values_len )
        #print(f'frame_values_len: {frame_values_len}')
        for value, k in zip(frame_values, range(len(frame_values))):
            val = float(value)
            i = int(k % Nx)
            j = int(k / Nx)
            image[i, j] = val
            if val < min_value:
                min_value = val
            elif val > max_value:
                max_value = val
    else:
        print(f'no frame: {frame_str_length}');
                


print(f'min_value: {min_value}, max_value: {max_value}')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure( figsize=(12,12) )

a = images[0]
#im = plt.imshow(a, interpolation='none', cmap='gray', aspect='auto', vmin=0, vmax=1)
im = plt.imshow(a, cmap='Greys', aspect='auto', vmin=min_value, vmax=max_value)

def animate_func(i):
    im.set_array(images[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               interval = 50, # in ms
                               
                               )

#anim.save('test_anim.avi', fps=30)
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