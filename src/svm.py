"""
Support vector machine code to classify responses of AlexNet.
    - type1 argument: path to directory containing images of type 1
    - type2 argument: path to directory containing images of type 2
    - others argument: path to directory containing images of all other types
"""
import argparse
import pickle
from os import walk

import caffe
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from env.env import *

caffe.set_mode_gpu()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--type1', default=research_root + 'images/eyes/normal/sm/', required=False,
                    help='Path to image files of type 1 (e.g. eye images)')
parser.add_argument('--type2', default=research_root + 'images/noses/normal/sm/', required=False,
                    help='Path to image files of type 2 (e.g. nose images)')
parser.add_argument('--others', default=research_root + 'images/others/sm/', required=False,
                    help='Path to other types of image files (e.g. not eye and not nose)')
parser.add_argument('--dump', help='Dump variables to files for fast loading', action='store_true')
parser.add_argument('--load-dump', dest='loaddump', action='store_true',
                    help='Load dumped variables, including datapoints, datalabels, and clf')
args = parser.parse_args()

imagenet_labels_filename = original_models_root + 'ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Net(
    original_models_root + 'bvlc_reference_caffenet/deploy.prototxt',
    original_models_root + 'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
    caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(ilsvrc_mean_file_path).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# List of fc6 responses of each image
datapoints = []
# List of labels of each image (1, 2 or 3)
datalabels = []
clf = svm.SVC()

pairs = [(args.type1, 1), (args.type2, 2), (args.others, 3)]


def load_image(image_path):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
    out = net.forward()


def predict():
    return clf.predict([net.blobs['fc6'].data[0].tolist()])


if not args.loaddump:

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
