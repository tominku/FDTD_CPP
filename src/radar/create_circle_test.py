import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
#import json
import ujson as json

print(f'{PIL.__version__}')
img_width = 100
img = Image.new("L", (img_width, img_width), 0)
draw = ImageDraw.Draw(img)
#draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))
#draw.
draw.circle((60, 60), 8, fill=255, width=0)

arr_img = np.array(img)
print(arr_img.shape)
print(arr_img)
arr_img_flatten = arr_img.ravel(order='F')
img.save("circle_image.png", "PNG")
#print(json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4))
output = {"width":img_width,
            "height":img_width, "data":arr_img_flatten.tolist()}
outfile_name = "data/material_circle.json"
with open(outfile_name, "w") as outfile:
    json.dump(output, outfile)
