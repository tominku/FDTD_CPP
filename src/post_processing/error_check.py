import numpy as np
import filecmp

path_cpu = "/home/minku/.data/output_cpu.txt"
path_gpu = "/home/minku/.data/output_gpu.txt"
output_cpu = open("/home/minku/.data/output_cpu.txt", "r")
output_gpu = open("/home/minku/.data/output_gpu.txt", "r")

result = filecmp.cmp(path_cpu, path_gpu)
print("cmp result: %d", result)

