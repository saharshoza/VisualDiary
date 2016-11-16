from images2gif import writeGif
from PIL import Image
import os
import glob
import sys
import re

path_to_images = PATH
out_gif = 

def generate_gif(change_point_list, path_to_images, sampling_rate_gif=15, filename):
	start_frame = 0
	gif_list_local = []
	gif_list_global = []
	images_list = [path_to_images+f for f in os.listdir(path_to_images) if re.search('jpg|JPG', f)]
	for change_point in change_point_list:
		gif_single = []
		for frame_index in range(start_frame, change_point):
			if (frame_index % sampling_rate_gif == 0):
				gif_single.append(images_list[frame_index])
		gif_list_local.append(gif_single)
		gif_list_global.extend(gif_single)
		start_frame = change_point
	images = [Image.open(fn) for fn in gif_list_global]
	writeGif(filename, images, duration=0.3, subRectangles=False)

if __name__ ==  "__main__":
	generate_gif(change_point_list, path_to_images, sampling_rate_gif, out_gif)