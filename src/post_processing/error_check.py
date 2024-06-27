import numpy as np
import filecmp

path_cpu = "/home/minku/.data/output_cpu.txt"
path_gpu = "/home/minku/.data/output_gpu.txt"
output_cpu = open("/home/minku/.data/output_cpu.txt", "r")
output_gpu = open("/home/minku/.data/output_gpu.txt", "r")

info_line = output_cpu.readline()
info_line2 = output_gpu.readline()
info = info_line.split(",")
Nx = int(info[0])
Ny = int(info[1])
steps = int(info[2])
logging_period = int(info[3])
N = Nx * Ny
print(f'Nx: {Nx}, Ny: {Ny}, Nt: {steps}, logging_period: {logging_period}')
min_value = 1e6
max_value = -1e6
frames = output_cpu.read().split(";")
frames2 = output_gpu.read().split(";")
for frame, frame2 in zip(frames, frames2):    
    frame_str_length = len(frame);
    if frame_str_length != 0:
        frame_values = frame.split(",");
        frame_values2 = frame2.split(",");
        frame_values_len = len(frame_values)
        assert( N == frame_values_len )
        #print(f'frame_values_len: {frame_values_len}')
        for value, value2, k in zip(frame_values, frame_values2, range(len(frame_values))):
            val = float(value)
            val2 = float(value2)

            print(f'error: {abs(val - val2)}')

            #i = int(k % Nx)
            #j = int(k / Nx)            
            if val < min_value:
                min_value = val
            elif val > max_value:
                max_value = val
    else:
        print(f'no frame: {frame_str_length}');

