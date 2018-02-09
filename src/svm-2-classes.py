import argparse
import os
import pickle
from os import walk

import caffe
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from env.env import *

caffe.set_mode_gpu()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--yes-images', dest='yesimages', default=research_root + 'images/flickr/eyes-yes/',
                    required=False)
parser.add_argument('--no-images', dest='noimages', default=research_root + 'images/flickr/flickr_other/',
                    required=False)

parser.add_argument('--dump', help='dump variables to files for fast loading', action='store_true')
parser.add_argument('--save-dump-to', dest='dumppath', help='path to save dump file', default='svm2c-data.dump',
                    required=False)
parser.add_argument('--load-dump', dest='loaddump', help='load dumped variables', action='store_true')

parser.add_argument('--layer', help='which layer in AlexNet to use for training and classification', default='pool5',
                    required=False)
# Default value means take the entire kernel
parser.add_argument('--kernel-size', dest='kernelsize', default=0, required=False,
                    help='what square size to take from the center of kernel to use for training and classification')
parser.add_argument('--skip-test', dest='skiptest', action='store_true')
args = parser.parse_args()

LAYER_TO_USE = args.layer

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Net(original_models_root + 'bvlc_reference_caffenet/deploy.prototxt',
                original_models_root + 'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(ilsvrc_mean_file_path).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

datapoints = []
datalabels = []
clf = svm.SVC()

pairs = [(args.yesimages, 1), (args.noimages, 2)]


def load_image(path):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
    out = net.forward()


def layer_to_svm_data(net):
    """
    Take the layer data from a Caffe instance, crop the filters if necessary,
    and return a 1D array for training and classification
    """
    layer_data = net.blobs[LAYER_TO_USE].data[0]
    if layer_data.size != len(layer_data):
        # Flatten it if the data is multidimensional
        # Also, the kernelsize parameter only makes sense here
        kernelsize = int(args.kernelsize)
        if kernelsize != 0:
            # Crop the center of the kernel of specified size
            # How many points to skip from the left and the top
            # Assuming layer_data[i] is a square
            skip = int((len(layer_data[0]) - kernelsize) / 2)
            layer_data = layer_data[:, skip:(skip+kernelsize), skip:(skip+kernelsize)]
        layer_data = np.reshape(layer_data, layer_data.size)

    return layer_data.tolist()


def predict():
    return clf.predict([layer_to_svm_data(net)])


def test_on_dir(path):
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            path = os.path.abspath(os.path.join(dirpath, filename))
            load_image(path)
            prediction = predict()
            print 'Prediction for ' + path + ': ' + prediction[0].astype('str')


print 'Using ' + LAYER_TO_USE + ' layer for training'
if args.dump:
    print 'Saving variables dump to ' + args.dumppath
if args.loaddump:
    print 'Loading variables dump from ' + args.dumppath

dump_filename = args.dumppath
if not args.loaddump:

    for pair in pairs:
        for (dirpath, dirnames, filenames) in walk(pair[0]):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                load_image(path)
                datapoints.append(layer_to_svm_data(net))
                datalabels.append(pair[1])

                print 'Processed ' + path + ' as type ' + str(pair[1])

    clf.fit(datapoints, datalabels)

    if args.dump:
        f = open(dump_filename, 'wb')
        pickle.dump([datapoints, datalabels, clf], f)
        f.close()
else:
    print 'Loading variables dump......'
    f = open(dump_filename)
    datapoints, datalabels, clf = pickle.load(f)
    f.close()
    print 'Loading finished.'

if not args.skiptest:
    num_data = 0
    num_correctly_classified = 0

    for i in range(len(datapoints)):
        num_data = num_data + 1
        prediction = clf.predict([datapoints[i]])
        if prediction[0].astype('str') == str(datalabels[i]):
            num_correctly_classified = num_correctly_classified + 1

        print 'Tested ' + str(num_data) + ', expected ' + str(datalabels[i]) + ', predicted ' + prediction[0].astype('str')

    print "Out of " + str(num_data) + " training examples, " + str(num_correctly_classified) + " were correctly classified"
