from constants import *
import argparse
from os import walk
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--yesimages', default=research_root + 'images/flickr/eyes-yes/', required=False)
parser.add_argument('--noimages', default=research_root + 'images/flickr/flickr_other/', required=False)
parser.add_argument('--dump', help='dump variables to files for fast loading', action='store_true')
parser.add_argument('--dumppath', help='path to save dump file', default='svm2c-data.dump', required=False)
parser.add_argument('--loaddump', help='load dumped variables', action='store_true')
parser.add_argument('--layer', help='which layer in AlexNet to use for training and classification', default='pool5', required=False)
parser.add_argument('--skiptest', action='store_true')
args = parser.parse_args()

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from sklearn import svm

LAYER_TO_USE = args.layer

print 'Using ' + LAYER_TO_USE + ' layer for training'
if args.dump:
    print 'Saving variables dump to ' + args.dumppath
if args.loaddump:
    print 'Loading variables dump from ' + args.dumppath

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

datapoints = []
datalabels = []
clf = svm.SVC()

pairs = [(args.yesimages, 1), (args.noimages, 2)]

def load_image(path):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
    out = net.forward()

def predict():
    layer_data = net.blobs[LAYER_TO_USE].data[0]
    if layer_data.size != len(layer_data):
        # Flatten it
        layer_data = np.reshape(layer_data, layer_data.size)

    return clf.predict([layer_data.tolist()])

def test_on_dir(path):
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            path = os.path.abspath(os.path.join(dirpath, filename))
            load_image(path)
            prediction = predict()
            print 'Prediction for ' + path + ': ' + prediction[0].astype('str')

dump_filename = args.dumppath
if args.loaddump == False:

    for pair in pairs:
        for (dirpath, dirnames, filenames) in walk(pair[0]):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                load_image(path)
                layer_data = net.blobs[LAYER_TO_USE].data[0]
                if layer_data.size != len(layer_data):
                    # Flatten it
                    layer_data = np.reshape(layer_data, layer_data.size)

                datapoints.append(layer_data.tolist())
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

if !args.skiptest:
    num_data = 0
    num_correctly_classified = 0

    for i in range(len(datapoints)):
        num_data = num_data + 1
        prediction = clf.predict([datapoints[i]])
        if prediction[0].astype('str') == str(datalabels[i]):
            num_correctly_classified = num_correctly_classified + 1

        print 'Tested ' + str(num_data) + ', expected ' + str(datalabels[i]) + ', predicted ' + prediction[0].astype('str')

    print "Out of " + str(num_data) + " training examples, " + str(num_correctly_classified) + " were correctly classified"