from constants import *
import argparse
from os import walk

parser = argparse.ArgumentParser(description='LOL')
parser.add_argument('--type1')
parser.add_argument('--type2')
parser.add_argument('--others')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

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

for (dirpath, dirnames, filenames) in walk(args.type1):
    for filename in filenames:
        path = dirpath + "/" + filename

        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
        out = net.forward()

        datapoints.append(net.blobs['fc6'].data[0].tolist())
        datalabels.append(0)

for (dirpath, dirnames, filenames) in walk(args.type2):
    for filename in filenames:
        path = dirpath + "/" + filename

        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
        out = net.forward()

        datapoints.append(net.blobs['fc6'].data[0].tolist())
        datalabels.append(1)

for (dirpath, dirnames, filenames) in walk(args.others):
    for filename in filenames:
        path = dirpath + "/" + filename

        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
        out = net.forward()

        datapoints.append(net.blobs['fc6'].data[0].tolist())
        datalabels.append(2)

from sklearn import svm
clf = svm.SVC()
clf.fit(datapoints, datalabels)

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('/mnt/software/research/images/cat-nose3.jpg'))
out = net.forward()