import os
import re
import sys
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
import av

def saliency_vector(features):
    feature_1 = np.vstack((np.zeros(1,features.shape[1]),features))
    feature_2 = np.vstack((features,np.zeros(1,features.shape[1])))
    diff_vector = feature_2 - feature_1
    return diff_vector

def frame_extractor(path_to_video,path_to_frames):
    container = av.open(path_to_video)
    video = next(s for s in container.streams if s.type == b'video')
    for packet in container.demux(video):
        for frame in packet.decode():
            frame.to_image().save(path_to_frames+'/frame-%06d.jpg' % frame.index)


class ExtractFeatures:
    def __init__(self,model_path,images_dir):
        self.model = model_path
        self.list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
    
    def create_graph(self):
        with gfile.FastGFile(self.model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    
    def get_layer_list(self):
        self.create_graph()
        with tf.Session() as sess:
            operation_list = sess.graph.get_operations()
            for op in operation_list:
                print op.name
                print op.values().get_shape()

    def extract_features(self,layer_name='pool_3:0',nb_features=2048):
        features = np.empty((len(self.list_images),nb_features))
        labels = []
        self.create_graph()
        with tf.Session() as sess:
            next_to_last_tensor = sess.graph.get_tensor_by_name(layer_name)
            for ind, image in enumerate(self.list_images):
                if (ind%100 == 0):
                    print('Processing %s...' % (image))
                if not gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)
                    continue
                image_data = gfile.FastGFile(image, 'rb').read()
                predictions = sess.run(next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
                features[ind,:] = np.squeeze(predictions)
        return features


if __name__ == "__main__":
	path_to_video = sys.argv[1]
	path_to_frames = sys.argv[2]
	model_dir = sys.argv[3]
	frame_extractor(path_to_video,path_to_frames)
	extract_features = ExtractFeatures(model_dir,images_dir)
	extract_features.create_graph()
	features = extract_features.extract_features()
	pickle.dump(features, open('features', 'wb'))
    diff_vector = saliency_vector(features)
    pickle.dump(diff_vector, open('diff_vector','wb'))