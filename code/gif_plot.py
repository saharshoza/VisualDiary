import os
import re
import sys
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import math
import pickle

# Image related imports
#import av # TODO: Remove this after porting to PIL
from PIL import Image
# from resizeimage import resizeimage
from images2gif import writeGif

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle

path_to_video = 'test.mp4'
data_dir = '../examples/0/input/'
model_dir = '../imagenet/classify_image_graph_def.pb'
results_dir = '../examples/0/results/'
images_dir = os.path.join(data_dir, 'frames/')
cnn_diff_dir = os.path.join(results_dir, 'cnn_diff_tmp')
pixel_diff_dir = os.path.join(results_dir, 'pixel_diff_tmp')
out_dir = os.path.join(results_dir, 'combined_frames_tmp')
path_to_gif = os.path.join(results_dir, 'gif')
path_to_gps = os.path.join(results_dir, 'gps_out/gps_out.npy')
path_to_ground_truth = os.path.join(data_dir, 'ground_truth/ground_truth.npy')
path_to_pixel_cnn_compare = os.path.join(results_dir,'pixel_cnn_compare/compare.png')
pickle_path = os.path.join(results_dir, 'vardump.pickle')

class PlotterOut:
    def __init__(self,differences,frames_for_video=150):
        self.frames_for_video = frames_for_video
        self.start_frame = np.random.choice(len(differences)-self.frames_for_video-1,1)[0]
    
    def save_difference_plots(self,differences, path, color):
        # 5 = 120 pixels
        differences = differences[self.start_frame:self.start_frame+self.frames_for_video]
        differences = differences / max(differences)
        plt.rcParams['figure.figsize'] = (11.25, 3.75) 
        plt.figure()
        num_frames = len(differences)
        image_idx_list = range(self.start_frame, self.start_frame + self.frames_for_video)
        for (list_idx,i) in enumerate(image_idx_list):
            plot_file = os.path.join(path, '%06d.png' % i)
            plt.plot(differences[:list_idx+1], color=color)
            plt.xlim([0, num_frames])
            plt.ylim([0, 1])
            plt.savefig(plot_file)
            
    def combine_images(self):
        # plt.rcParams['figure.figsize'] = (10, 5)
        image_paths = sorted(os.listdir(images_dir))
        num_images = self.frames_for_video
        image_idx_list = range(self.start_frame, self.start_frame + self.frames_for_video)
        for (list_idx,i) in enumerate(image_idx_list):
            if list_idx % 50 == 0:
                print('Now processing', list_idx, 'out of', num_images)
            im_file = os.path.join(images_dir, image_paths[i])
            cnn_diff_file = os.path.join(cnn_diff_dir, '%06d.png' % i)
            pixel_diff_file = os.path.join(pixel_diff_dir, '%06d.png' % i)
            out_file = os.path.join(out_dir, '%06d.png' % i)
            cnn_diff_im = np.asarray(Image.open(cnn_diff_file))
            pixel_diff_im = np.asarray(Image.open(pixel_diff_file))
            diff_im = np.vstack([cnn_diff_im, pixel_diff_im])
            diff_im = diff_im[:,:,0:3]
            im = Image.open(im_file)
            im = im.resize((diff_im.shape[1],diff_im.shape[0]),Image.ANTIALIAS)
            im = np.asarray(im)
            combined = np.hstack([im, diff_im])
            combined_image = Image.fromarray(combined)
            #combined_image.show()
            combined_image.save(os.path.join(out_dir, '%06d.png' % list_idx), 'PNG')    
    
    @staticmethod
    def get_smoothed_list(list_to_smooth, window_size = 10):
        num_elements = len(list_to_smooth)
        smoothed_list = []
        cur_sum = sum(list_to_smooth[:window_size])
        smoothed_list.append(cur_sum * 1.0 / window_size)
        for i in xrange(window_size, num_elements):
            cur_sum -= list_to_smooth[i-window_size]
            cur_sum += list_to_smooth[i]
            smoothed_list.append(cur_sum * 1.0 / window_size)
        return smoothed_list
    
    @staticmethod
    def plot_differences(feature_differences, pixel_differences, path_to_pixel_cnn_compare):
        feature_differences = feature_differences / max(feature_differences)
        pixel_differences = pixel_differences / max(pixel_differences)
        plt.plot(feature_differences, label='Feature Differences')
        plt.plot(pixel_differences, label='Pixel Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Difference')
        plt.title('Comparison of Change Point between CNN and Raw Pixel')
        plt.legend()
        plt.savefig(path_to_pixel_cnn_compare)
        
    @staticmethod   
    def generate_gif(change_point_list, path_to_frames, path_to_gif):
        gif_list = []
        all_images_list = [path_to_frames + f for f in os.listdir(path_to_frames) if re.search('jpg|JPG', f)]
        gif_list_of_lists = [[change_point,change_point+(change_point_list[change_point_index+1]-change_point_list[change_point_index])/3,change_point+2*(change_point_list[change_point_index+1]-change_point_list[change_point_index])/3] for (change_point_index,change_point) in enumerate(change_point_list) if change_point_index<len(change_point_list)-1]
        for gif_segment in gif_list_of_lists:
            gif_list.extend(gif_segment)
        selected_gif_images = [Image.open(all_images_list[gif_points]) for gif_points in gif_list]
        writeGif(path_to_gif, selected_gif_images, duration=0.3, subRectangles=False)

if __name__ == "__main__":

    variables = pickle.load(open(pickle_path,'r'))
    cnn_differences = variables[0]
    pixel_differences = variables[1]
    cnn_change_points = variables[2]
    pixel_change_points = variables[3]
    vid_gps_changepoints = variables[4]
    cnn_penalty = variables[4]
    vid_gps_penalty = variables[5]

    # Instantiate class to give plots, gif and video
    plotter_out = PlotterOut(cnn_differences)
    
    # Generate gif using cnn, pixel and video + gps
    plotter_out.generate_gif(cnn_change_points, images_dir, path_to_gif+'/cnn.gif')
    plotter_out.generate_gif(pixel_change_points, images_dir, path_to_gif+'/pixel.gif')
    plotter_out.generate_gif(vid_gps_changepoints, images_dir, path_to_gif+'/gps_vid.gif')
    
    # Plot the differences in saliency identified by cnn and features separately
    plotter_out.plot_differences(plotter_out.get_smoothed_list(cnn_differences), plotter_out.get_smoothed_list(pixel_differences),path_to_pixel_cnn_compare)
    
    # This is for generating video showing the cnn superiority over pixel differences.
    # Uncomment this for all but demo video sequence (200 frames at max)
    plotter_out.save_difference_plots(cnn_differences, cnn_diff_dir, '#ff6666')
    plotter_out.save_difference_plots(pixel_differences, pixel_diff_dir, '#79cdcd')
    plotter_out.combine_images()
