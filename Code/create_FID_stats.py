#!/usr/bin/env python3

import os
import glob
import numpy as np
import fid_new as fid
from scipy.misc import imread
import tensorflow as tf

########
# PATHS
########

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/Datasets/CIFAR10_images/train')
parser.add_argument('--inception_path', default='/models/Inception')
parser.add_argument('--output_path', default='CIFAR_fid_stats.npz')
parser.add_argument('--batch_size', default=100)
param = parser.parse_args()

# Add local variable to folders so we work in the local drive
param.data_path = os.environ["SLURM_TMPDIR"] + param.data_path
param.output_path = os.environ["SLURM_TMPDIR"] + param.output_path
param.inception_path = os.environ["SLURM_TMPDIR"] + param.inception_path

data_path = param.data_path # set path to training set images
output_path = param.output_path # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = param.inception_path
print(inception_path)
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
image_list.extend(glob.glob(os.path.join(data_path, '*.png')))
#images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
#print("%d images found and loaded" % len(images))
print("%d images found and loaded" % len(image_list))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics_from_files(image_list, sess, batch_size=param.batch_size)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")