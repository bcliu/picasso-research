from constants import *
import argparse
from os import walk
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototext',
                caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# Dimensions: 224x224, with 3 channels. Batch size 1
# NOTE: maybe can use batching to speed up processing?
net.blobs['data'].reshape(1, 3, 224, 224)

def load_image(path):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'][0].argmax()))

    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print labels[top_k]

load_image('/home/chenl/research/images/bulls/bull2.jpg')

# Find better way to write it to distribute more evenly
def sample(width, height, number):
    mat = [[False for x in range(width)] for x in range(width)]
    prob_true = number * 1.0 / width / height
    for x in range(width):
        for y in range(height):
            mat[y][x] = False if np.random.random() < prob_true else True

    return mat


