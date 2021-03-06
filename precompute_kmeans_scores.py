import argparse
import pickle
import sys

import caffe
import matplotlib
import numpy as np

from env.env import *

caffe.set_mode_gpu()
matplotlib.use('Agg')  # For saving images to file

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ld', '--layer_dump', required=True)
parser.add_argument('-cd', '--clusters_dump', required=True)
parser.add_argument('-o', '--output_dump', required=True)
args = parser.parse_args()

net = caffe.Classifier(
    original_models_root + 'vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
    original_models_root + 'vgg16/VGG_ILSVRC_16_layers.caffemodel',
    mean=np.load(ilsvrc_mean_file_path).mean(1).mean(1),
    channel_swap=(2, 1, 0),
    raw_scale=255,
    image_dims=(224, 224))
net.blobs['data'].reshape(1, 3, 224, 224)

print 'Loading layer dump file from', args.layer_dump
f = open(args.layer_dump)
[_, layer, _, _, _, _, vectors, vec_origin_file, vec_location] = pickle.load(f)
f.close()
print 'Finished.'

print 'Loading clusters dump from', args.clusters_dump
f = open(args.clusters_dump)
dumped = pickle.load(f)
kmeans_scores = []
if len(dumped) == 3:
    n_clusters, kmeans_obj, predicted = dumped
elif len(dumped) == 4:
    n_clusters, kmeans_obj, predicted, kmeans_scores = dumped
f.close()
print 'Finished.'

n_vectors = len(vectors)

if len(kmeans_scores) == 0:
    print 'Calculating kmeans scores...\n'
    for vec_i in range(n_vectors):
        kmeans_scores.append(kmeans_obj.score(vectors[vec_i].reshape(1, -1)))
        sys.stdout.write("\rFinished: %f%%" % (vec_i * 100.0 / n_vectors))
        sys.stdout.flush()
    print '\nDone. Writing to updated dump to', args.output_dump
    f = open(args.output_dump, 'wb')
    pickle.dump([n_clusters, kmeans_obj, predicted, kmeans_scores], f)
    f.close()
    print 'Finished.'
else:
    print 'Nothing to be done: kmeans_score present'
