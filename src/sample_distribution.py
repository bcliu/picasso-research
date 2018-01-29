# Idea: take a few images (e.g. 50). Based on the distribution of eyes neural response in the high dimensional space,
# randomly sample new responses in the hypercolumn (or neighboring hypercolumns) to generate new images

from env.env import *
import argparse
import os
from os import walk
import caffe
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import get_receptive_field as rf

os.environ['GLOG_minloglevel'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--images', '-i', required=True)
parser.add_argument('--layer', '-l', default='conv3_3')
parser.add_argument('--sample_fraction', '-f', default=1.0, type=float)
parser.add_argument('--n_clusters', '-n', default=32, type=int)
args = parser.parse_args()

caffe.set_mode_gpu()

# TODO: how about feeding some eye-only patches again?


def sample(width, height, number):
    prob_true = number * 1.0 / width / height
    return np.random.rand(height, width) < prob_true


print 'Loading VGG model...'
net = caffe.Classifier(
    original_models_root + 'vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
    trained_models_root + 'vgg16_models/flickr_40k_iter_21000.caffemodel',
    mean=np.load(ilsvrc_mean_file_path).mean(1).mean(1),
    channel_swap=(2, 1, 0),
    raw_scale=255,
    image_dims=(224, 224))

net.blobs['data'].reshape(1, 3, 224, 224)


def load_image(image_path):
    net.predict([caffe.io.load_image(image_path)], oversample=False)


sample_mask = None
vectors = []
vec_origin_file = []
vec_location = []

for (dirpath, dirnames, filenames) in walk(os.path.abspath(args.images)):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        print 'Processed', path
        load_image(path)

        response = net.blobs[args.layer].data[0]
        num_feature_maps = len(response)
        height_feature_map = len(response[0])
        width_feature_map = len(response[0])

        if sample_mask is None:
            sample_mask = sample(width_feature_map, height_feature_map,
                                 args.sample_fraction * width_feature_map * height_feature_map)

        for y in range(height_feature_map):
            for x in range(width_feature_map):
                if sample_mask[y][x]:
                    vec_origin_file.append(path)
                    vectors.append(response[:, y, x].copy())
                    vec_location.append((x, y))

print 'Got', len(vectors), 'vectors randomly sampled'

print 'Clustering...'
n_restarts = 10
kmeans_obj = KMeans(init='k-means++', n_clusters=args.n_clusters, n_init=n_restarts)
predicted = kmeans_obj.fit_predict(vectors)
kmeans_scores = []
print 'Done.'


# TODO: show a screen of visualized clusters, then ask to select one cluster, then
# TODO: try to do PCA, show a distribution on each dimension, etc. Do a kernel density est.


def get_original_patch_of_vec(vec_id):
    this_vec_location = vec_location[vec_id]
    rec_field = rf.get_receptive_field(args.layer, this_vec_location[0], this_vec_location[1])
    load_image(vec_origin_file[vec_id])
    im = net.transformer.deprocess('data',
                                   net.blobs['data'].data[0][:, rec_field[1]:(rec_field[3]+1), rec_field[0]:(rec_field[2]+1)])
    return im


def get_top_n_in_cluster(cluster_i, n):
    scores = []

    for vec_id in range(len(vectors)):
        if predicted[vec_id] == cluster_i:
            scores.append((vec_id, kmeans_obj.score(vectors[vec_id].reshape(1, -1))))

    scores.sort(key=lambda tup: -tup[1])
    if n == -1:
        return scores
    return scores[0:n]


def view_n_from_clusters(from_cluster, to_cluster, n_each, save_plots=False):
    fig = plt.figure()
    fig_id = 1

    for i in range(from_cluster, to_cluster+1):
        scores = get_top_n_in_cluster(i, n_each)
        for (vec_id, score) in scores:
            print 'Vector #', vec_id, 'in cluster #', i, ', score:', score
            fig.add_subplot(to_cluster - from_cluster + 1, n_each, fig_id)
            plt.imshow(get_original_patch_of_vec(vec_id))

            fig_id = fig_id + 1

            plt.axis('off')

    if save_plots:
        plt.savefig(os.path.join(args.save_plots_to, args.layer + '_' + 'clusters' + str(from_cluster) + 'to' + str(to_cluster) + '.png'))
    else:
        plt.show()
