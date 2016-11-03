# This code will pick all the .jpg and .png files from the current directory and then create a gif of them

from images2gif import writeGif
from PIL import Image
import os
import glob

file_types = ["*.jpg", "*.png"]
images_captured = []

for iterator in file_types:
    images_captured.extend(glob.glob(iterator))

images = [Image.open(fn) for fn in images_captured]
size = (600,350)

for im in images:
    im.thumbnail(size, Image.ANTIALIAS)

filename = "doge.gif"
writeGif(filename, images, duration=0.3, subRectangles=False)
