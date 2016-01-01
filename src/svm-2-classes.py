from constants import *
import argparse
from os import walk
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--yesimages', default=research_root + 'images/eyes/normal/sm/', required=False)
parser.add_argument('--noimages', default=research_root + 'images/others/sm/', required=False)
parser.add_argument('--dump', help='dump variables to files for fast loading', action='store_true')
parser.add_argument('--loaddump', help='load dumped variables', action='store_true')
args = parser.parse_args()

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from sklearn import svm

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
    return clf.predict([net.blobs['fc6'].data[0].tolist()])

dump_filename = 'svm-2-classes-data.dump'
if args.loaddump == False:

    for pair in pairs:
        for (dirpath, dirnames, filenames) in walk(pair[0]):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                load_image(path)
                datapoints.append(net.blobs['fc6'].data[0].tolist())
                datalabels.append(pair[1])

                print 'Processed ' + path + ' as type ' + str(pair[1])

    clf.fit(datapoints, datalabels)

    if args.dump:
        f = open(dump_filename, 'wb')
        pickle.dump([datapoints, datalabels, clf], f)
        f.close()
else:
    f = open(dump_filename)
    datapoints, datalabels, clf = pickle.load(f)
    f.close()

num_data = 0
num_correctly_classified = 0

for pair in pairs:
    for (dirpath, dirnames, filenames) in walk(pair[0]):
        for filename in filenames:
            path = os.path.abspath(os.path.join(dirpath, filename))
            num_data = num_data + 1
            load_image(path)
            prediction = predict()
            if prediction[0].astype('str') == str(pair[1]):
                num_correctly_classified = num_correctly_classified + 1

print "Out of " + str(num_data) + " training examples, " + str(num_correctly_classified) + " were correctly classified"
