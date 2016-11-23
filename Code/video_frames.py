import glob
import sys
import re
import time
import datetime
import ntpath
import os
import imageio
from PIL import Image
from images2gif import writeGif
import av

#all independent so far. so didn't make class

'''
Makes movie(gif) from all frames given
'''
def make_movie(folder_frames, movie_name=None):
	filenames = os.listdir(folder_frames)
	if movie_name is None:
		ts = time.time()
		movie_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
	with imageio.get_writer(movie_name+'.gif', mode='I') as writer:
	    for filename in filenames:
	        image = imageio.imread(folder_frames + '/' + filename)
	        writer.append_data(image)

def extract_frames(movie_path,folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
	container = av.open(movie_path)
	video = next(s for s in container.streams if s.type == b'video')
	for packet in container.demux(video):
		for frame in packet.decode():
			frame.to_image().save(folder+'/%04d.jpg' % frame.index)
			
def generate_gif(path_to_images, change_point_list, filename, sampling_rate_gif=15):
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

