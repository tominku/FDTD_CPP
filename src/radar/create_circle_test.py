import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
#import json
import ujson as json
import numpy.linalg as LA

print(f'{PIL.__version__}')
img_width = 100
img = Image.new("L", (img_width, img_width), 0)
draw = ImageDraw.Draw(img)
#draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))
#draw.
radius = 4
draw.circle((20, 40), radius, fill=50, width=0)
#draw.rectangle((30, 40, 60, 60), fill=50, width=0)
pixel_value_to_eps = {50: 30, 30: 11.68}

vec = np.array([0.2, 0.8]) - np.array([0.4, 0.2])
print(f'distance: {LA.norm(vec)} m')

arr_img = np.array(img)
print(arr_img.shape)
print(arr_img)
arr_eps_r = np.ones(arr_img.shape) 
print(arr_eps_r)
key = 50
arr_eps_r[arr_img == key] = pixel_value_to_eps[key]
print(arr_eps_r)


arr_img_flatten = arr_img.ravel(order='F')
arr_eps_r_flatten = arr_eps_r.ravel(order='F')
img.save("circle_image.png", "PNG")
#print(json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4))
output = {"width":img_width,
            "height":img_width, "data":arr_eps_r_flatten.tolist()}
outfile_name = "data/material_circle.json"
#outfile_name = "data/material_rect.json"
with open(outfile_name, "w") as outfile:
    json.dump(output, outfile)
