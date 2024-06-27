import numpy as np

output = open("/home/minku/.data/output.txt", "r")
info_line = output.readline()

info = info_line.split(",")
Nx = int(info[0])
Ny = int(info[1])
images = []
steps = int(info[2])
print(f'Nx: {Nx}, Ny: {Ny}, Nt: {steps}')

frames = output.read().split("\n*\n")
print(f'# frames: {len(frames)}')
assert(steps == len(frames))
#print(frames[0])
min_value = 1e6
max_value = -1e6
for frame in frames:
    image = np.zeros((Nx, Ny))
    images.append(image)
    lines = frame.split("\n")    
    for line, i in zip(lines, range(len(lines))):
        values = line.split(",")
        for value, j in zip(values, range(len(values))):
            val = float(value)
            image[i, j] = val
            if val < min_value:
                min_value = val
            elif val > max_value:
                max_value = val

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