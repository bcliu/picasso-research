from env.env import *
import argparse
from os import walk

import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--type1', default=research_root + 'images/eyes/normal/sm/', required=False)
parser.add_argument('--type2', default=research_root + 'images/noses/normal/sm/', required=False)
parser.add_argument('--others', default=research_root + 'images/others/sm/', required=False)
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

pairs = [(args.type1, 1), (args.type2, 2), (args.others, 3)]

def load_image(path):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))
    out = net.forward()

def predict():
    return clf.predict([net.blobs['fc6'].data[0].tolist()])

if args.loaddump == False:

    for pair in pairs:
        for (dirpath, dirnames, filenames) in walk(pair[0]):
            for filename in filenames:
                path = dirpath + filename
                load_image(path)
                datapoints.append(net.blobs['fc6'].data[0].tolist())
                datalabels.append(pair[1])

                print 'Processed ' + path + ' as type ' + str(pair[1])

    clf.fit(datapoints, datalabels)

    if args.dump:
        f = open('svm-data.dump', 'wb')
        pickle.dump([datapoints, datalabels, clf], f)
        f.close()
else:
    f = open('svm-data.dump')
    datapoints, datalabels, clf = pickle.load(f)
    f.close()

num_data = 0
num_correctly_classified = 0

for pair in pairs:
    for (dirpath, dirnames, filenames) in walk(pair[0]):
        for filename in filenames:
            path = dirpath + filename
            num_data = num_data + 1
            load_image(path)
            prediction = predict()
            if prediction[0].astype('str') == str(pair[1]):
                num_correctly_classified = num_correctly_classified + 1

print "Out of " + str(num_data) + " training examples, " + str(num_correctly_classified) + " were correctly classified"