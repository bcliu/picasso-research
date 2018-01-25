# Sort images (by assigning new filenames) by the activation of a specific neuron in fc8 layer
# or prob layer (containing 1000 classes)

from env.env import *
import argparse
import os
import sys
from os import walk
import numpy as np
import shutil

os.environ['GLOG_minloglevel'] = '2'

sys.path.insert(0, caffe_root + 'python')
import caffe
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images_dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-m', '--alexnet_model', required=True)
parser.add_argument('-l', '--layer', required=True)  # fc8 or prob
parser.add_argument('-n', '--neuron', default=643, type=int)  # default to the mask/face class
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       os.path.abspath(args.alexnet_model),
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(227, 227))

net.blobs['data'].reshape(1, 3, 227, 227)

# Array of tuples of image path and activation value (or whatever to sort by)
images_vals = []


def load_image(path):
    net.predict([caffe.io.load_image(path)], oversample=False)

for (dirpath, dirnames, filenames) in walk(os.path.abspath(args.images_dir)):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        load_image(path)
        val = net.blobs[args.layer].data[0][args.neuron]
        images_vals.append((path, val))

        sys.stdout.write("\x1b[2K\rProcessed: %s" % path)
        sys.stdout.flush()
print '\n'

images_vals.sort(key=lambda tup: tup[1], reverse=True)

# Start renaming files
for i in xrange(len(images_vals)):
    path, val = images_vals[i]
    name, ext = os.path.splitext(path)
    shutil.copy(path, os.path.abspath(os.path.join(args.output_dir, str(i) + ext)))
