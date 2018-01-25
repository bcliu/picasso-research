# Plot response curve in fc8 and prob layers of a certain neuron. Takes two types of images
# e.g. face/non-face images

from env.env import *
import argparse
from os import walk
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Set Caffe output level to Warnings
os.environ['GLOG_minloglevel'] = '2'

sys.path.insert(0, caffe_root + 'python')
import caffe
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dirs', nargs='+')
parser.add_argument('-id', '--input_dump', required=False)
parser.add_argument('-n', '--neuron', required=False, default=643, type=int)
parser.add_argument('-m', '--alexnet_model')
parser.add_argument('-o', '--save_dump_to', required=False, default=None)
parser.add_argument('-b', '--histogram_bins', type=int, default=50, required=False)
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--gpu', required=False, type=int, default=0)
args = parser.parse_args()

# Array of arrays: each element array i contains activations of all input images in
# directory i
fc8_activations = []
prob_activations = []

# Load from directory if no dump is specified
if args.input_dump is None:
    net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                           caffe_root + 'models/bvlc_reference_caffenet/' + args.alexnet_model,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(227, 227))
    net.blobs['data'].reshape(1, 3, 227, 227)

    for dir_i in range(len(args.input_dirs)):
        input_dir = args.input_dirs[dir_i]
        print 'Loading from type ' + str(dir_i) + ' images:', input_dir
        fc8_activations.append([])
        prob_activations.append([])

        loaded = 0
        for (dirpath, dirnames, filenames) in walk(input_dir):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                loaded += 1
                sys.stdout.write("\x1b[2K\rLoaded: %d (%s)" % (loaded, path))
                sys.stdout.flush()

                net.predict([caffe.io.load_image(path)], oversample=False)
                fc8_activations[dir_i].append(net.blobs['fc8'].data[0][args.neuron])
                prob_activations[dir_i].append(net.blobs['prob'].data[0][args.neuron])

        print '\n'

    # Save dump to file if specified
    if args.save_dump_to is not None:
        out_f = open(args.save_dump_to, 'wb')
        print 'Saving dump to', args.save_dump_to
        pickle.dump([fc8_activations, prob_activations], out_f)
else:
    # If dump file is specified, load arrays from it
    dump_f = open(args.input_dump)
    fc8_activations, prob_activations = pickle.load(dump_f)
    dump_f.close()

if not args.no_plot:
    num = len(fc8_activations)
    colors = ['blue', 'red', 'green', 'yellow', 'black']
    print 'Normalized fc8 activations:'
    for i in range(num):
        n, bins, patches = plt.hist(fc8_activations[i], args.histogram_bins, normed=1, facecolor=colors[i], alpha=0.5)
    plt.show()

    print 'Not normalized fc8 activations:'
    for i in range(num):
        n, bins, patches = plt.hist(fc8_activations[i], args.histogram_bins, normed=0, facecolor=colors[i], alpha=0.5)
    plt.show()

    print 'Normalized prob layer activations:'
    for i in range(num):
        n, bins, patches = plt.hist(prob_activations[i], args.histogram_bins * 10, normed=1, facecolor=colors[i], alpha=0.5)
    plt.show()

    print 'Not normalized prob layer activations:'
    for i in range(num):
        n, bins, patches = plt.hist(prob_activations[i], args.histogram_bins * 10, normed=0, facecolor=colors[i], alpha=0.5)
    plt.show()
