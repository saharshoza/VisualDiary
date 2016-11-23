import numpy as np
import random

def get_motion_boundaries(gps_stationary_intervals):
	boundaries = []
	for interval in gps_stationary_intervals:
		boundaries.append(interval[0])
		boundaries.append(interval[1])
	return boundaries

def get_top_k(differences, k, min_gap):
	ascending_indices = np.argsort(differences)
	chosen = []
	for i in reversed(ascending_indices):
		if len(chosen)==k:
			break
		if ((len(chosen)>0) and (abs(i-chosen[-1])>=min_gap)) or (len(chosen)==0):
			chosen.append(i)
	return chosen

def merge_changepoints(gps_stationary_intervals, image_differences):
	
	boundaries = get_motion_boundaries(gps_stationary_intervals)
	#adding last frame index
	boundaries.append(len(image_differences)-1)

	changepoints = []
	start = 0
	for end in boundaries:
		indices_in_window = get_top_k(image_differences[start:end+1], 2, 4)
		changepoints += [a+start for a in indices_in_window]

		start = end
	return changepoints

# differences = range(1000)
# # random.shuffle(differences)
# print merge_changepoints([(150,360),(450,750)], differences)