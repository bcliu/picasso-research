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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--images', default=research_root + 'images/flickr/eyes-yes/', required=False)
parser.add_argument('--layer', default='conv2_1', required=False)
parser.add_argument('--sample_fraction', default=0.3, required=False)
args = parser.parse_args()

print 'Loading images from ' + args.images
print 'Sampling ' + str(args.sample_fraction) + ' of responses from layer ' + args.layer

import caffe

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
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


def load_image(path, echo=True):
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


## NOTE: Resize every image to 224x224, or resize but keep original ratio? Does it affect things?
## Without resizing, feeding into data, does it crop or will resize?

sample_mask = []

# Array of vectors
vectors = []
# Array of file paths
vec_origin_file = []
# Array of (arrays of (x, y))
vec_location = []

# Loop through every image in the given directory
for (dirpath, dirnames, filenames) in walk(args.images):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        load_image(path, False)

        response = net.blobs[args.layer].data[0]
        num_responses = len(response)
        height_response = len(response[0])
        width_response = len(response[0][0])

        if len(sample_mask) == 0:
            # sample_mask not initialized yet; sample new
            print str(num_responses) + ' filters of ' + str(height_response) + 'x' + str(width_response)

            sample_mask = sample(width_response, height_response, float(args.sample_fraction))

        for y in range(height_response):
            for x in range(width_response):
                if sample_mask[y][x]:
                    ## NOTE: DOUBLE CHECK IF FIRST IS Y SECOND IS X, corresponding to images
                    vectors.append(response[:, y, x])
                    vec_origin_file.append(path)
                    vec_location.append((x, y))