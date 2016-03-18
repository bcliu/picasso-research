from constants import *
import argparse
from os import walk
import os

import math
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
import pickle

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import get_receptive_field as rf

from matplotlib.patches import Rectangle

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Find better way to write it to distribute more evenly
def sample(width, height, number):
    mat = [[False for x in range(width)] for x in range(width)]
    prob_true = number * 1.0 / width / height
    for x in range(width):
        for y in range(height):
            mat[y][x] = True if np.random.random() < prob_true else False

    return mat

parser = argparse.ArgumentParser(description='')
parser.add_argument('--images', default=research_root + 'images/flickr/eyes-yes/', required=False)
parser.add_argument('--layer', default='conv4_1', required=False)
parser.add_argument('--sample_fraction', default=0.3, required=False)
parser.add_argument('--n_clusters', default=32, required=False)
parser.add_argument('--center_only_path', default=None, required=False)
parser.add_argument('--center_only_neuron_x', default=None, required=False)
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--load_dump_from', default=None, required=False)
parser.add_argument('--save_dump_to', default=None, required=False)
parser.add_argument('--save_plots_to', default=None, required=False)

args = parser.parse_args()

if args.save_plots_to is not None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

print 'Loading images from ' + args.images
print 'Sampling ' + str(args.sample_fraction) + ' of responses from layer ' + args.layer

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

caffe.set_device(int(args.gpu))
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2,1,0))
net = caffe.Classifier(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
        caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2, 1, 0),
        raw_scale=255,
        image_dims=(224, 224))

# Dimensions: 224x224, with 3 channels. Batch size 1
# NOTE: maybe can use batching to speed up processing?
net.blobs['data'].reshape(1, 3, 224, 224)


def load_image(path, echo=True):
    net.predict([caffe.io.load_image(path)], oversample=False)
    #if echo:
    #    print("Predicted class is #{}.".format(out['prob'][0].argmax()))

    #    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #    print labels[top_k]


## NOTE: Resize every image to 224x224, or resize but keep original ratio? Does it affect things?
## Without resizing, feeding into data, does it crop or will resize?

sample_mask = []

# Array of vectors
vectors = []
# Array of file paths
vec_origin_file = []
# Array of (arrays of (x, y))
vec_location = []

if args.load_dump_from is not None:
    print 'Loading dump file from', args.load_dump_from
    f = open(args.load_dump_from)
    args_dump, sample_mask, vectors, vec_origin_file, vec_location, n_clusters, kmeans_obj, predicted = pickle.load(f)
    if hasattr(args, 'load_dump_from'):
        args_dump.load_dump_from = args.load_dump_from
    if hasattr(args, 'save_dump_to'):
        args_dump.save_dump_to = args.save_dump_to
    args = args_dump
    f.close()
    print 'Finished loading dump.'
else:
    # Loop through every image in the given directory
    for (dirpath, dirnames, filenames) in walk(args.images):
        for filename in filenames:
            path = os.path.abspath(os.path.join(dirpath, filename))
            print 'Processed', path
            load_image(path, False)

            response = net.blobs[args.layer].data[0]
            num_responses = len(response)
            height_response = len(response[0])
            width_response = len(response[0][0])

            if len(sample_mask) == 0:
                # sample_mask not initialized yet; sample new
                print str(num_responses) + ' filters of ' + str(height_response) + 'x' + str(width_response)

                sample_mask = sample(width_response, height_response,
                        float(args.sample_fraction) * width_response * height_response)

            for y in range(height_response):
                for x in range(width_response):
                    if sample_mask[y][x]:
                        ## NOTE: DOUBLE CHECK IF FIRST IS Y SECOND IS X, corresponding to images
                        vectors.append(response[:, y, x].copy())
                        vec_origin_file.append(path)
                        vec_location.append((x, y))


    print 'Got', len(vectors), 'vectors randomly sampled'
    # Load the images of which only the center patches will be used
    if args.center_only_path is not None:
        print 'Loading images of which center patches will be used'
        location_to_pick = int(args.center_only_neuron_x)
        print 'Center neuron x and y coordinate:', location_to_pick

        for (dirpath, dirnames, filenames) in walk(args.center_only_path):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                print 'Processed', path
                load_image(path, False)

                response = net.blobs[args.layer].data[0]
                vectors.append(response[:, location_to_pick, location_to_pick].copy())
                vec_origin_file.append(path)
                vec_location.append((location_to_pick, location_to_pick))

    # TRY PCA, ICA AS WELL!!!
    # if this method works, continue to explore how to make your initial method work as well

    n_clusters = int(args.n_clusters)
    n_restarts = 10
    kmeans_obj = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_restarts)
    predicted = kmeans_obj.fit_predict(vectors)

    if args.save_dump_to is not None:
        print 'Saving dump to', args.save_dump_to
        f = open(args.save_dump_to, 'wb')
        pickle.dump([args, sample_mask, vectors, vec_origin_file, vec_location, n_clusters, kmeans_obj, predicted], f)
        f.close()
        print 'Finished saving dump'

print 'Got', len(vectors), 'vectors in total for clustering'

##### PCA
pca = PCA(n_components=80)
#vectors_trans = pca.fit_transform(vectors)
#print pca.explained_variance_ratio_
#predicted = kmeans_obj.fit_predict(vectors_trans)


#########

def get_top_n_in_cluster(cluster_i, n):
    scores = []
    for vec_id in range(len(vectors)):
        if predicted[vec_id] == cluster_i:
            scores.append((vec_id, kmeans_obj.score(vectors[vec_id].reshape(1, -1))))

    scores.sort(key=lambda tup: -tup[1])
    if n == -1:
        return scores
    return scores[0:n]


## distance_threshold: require a vector to be smaller than distance of distance_threshold * num of vectors in cluster
##                     for it to be considered as in this cluster
## NOTE: CHECK THIS PART OF THRESHOLDING, TESTING
def find_patches_in_cluster(cluster_i, image_path, dist_thres_percentage=1.0):
    score_thres = 0.0
    cluster_scores = get_top_n_in_cluster(cluster_i, -1) # All score values in this cluster
    cluster_scores = [score for (vec_id, score) in cluster_scores]
    # Figure out exact value of distance threshold given the percentage
    score_thres = cluster_scores[int(math.floor(len(cluster_scores) * dist_thres_percentage)) - 1]
    print 'Limiting cluster score to smaller than', score_thres

    # Given an image, find the patches in the image that have responses in the given cluster
    load_image(image_path, False)
    dim_filter = len(net.blobs[args.layer].data[0][0])
    
    plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0]))
    axis = plt.gca()

    total_patches_found = 0
    patches_ignored = 0 # Number of patches that are in the cluster but ignored due to score

    for y in range(dim_filter):
        for x in range(dim_filter):
            hypercolumn = net.blobs[args.layer].data[0][:,y,x].copy().reshape(1, -1)
            prediction = kmeans_obj.predict(hypercolumn)
            if prediction == cluster_i:
                total_patches_found = total_patches_found + 1
                if kmeans_obj.score(hypercolumn) < score_thres:
                    patches_ignored = patches_ignored + 1
                else:
                    rec_field = rf.get_receptive_field(args.layer, x, y)
                    #### NOTE: VERIFY THAT YOU GOT X AND Y RIGHT AGAIN!!!
                    axis.add_patch(Rectangle((rec_field[0], rec_field[1]),
                        rec_field[2] - rec_field[0] + 1,
                        rec_field[3] - rec_field[1] + 1,
                        fill=False, edgecolor="red"))
                    #plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0][:,rec_field[1]:(rec_field[3]+1),rec_field[0]:(rec_field[2]+1)]))

    print 'Found', total_patches_found, 'patches in total,', patches_ignored, 'ignored due to distance'
    plt.show()

def view_nth_in_cluster(cluster_i, i):
    num_in_cluster_seen = 0
    for vec_id in range(len(vectors)):
        if predicted[vec_id] == cluster_i:
            if num_in_cluster_seen == i:
                print 'Showing the ' + str(vec_id) + 'th element in vectors array, at ', vec_location[vec_id]
                rec_field = rf.get_receptive_field(args.layer, vec_location[vec_id][0], vec_location[vec_id][1])
                print 'Receptive field: ', rec_field
                
                # CHECK IF YOU GET X AND Y RIGHT
                print 'From file:', vec_origin_file[vec_id]
                load_image(vec_origin_file[vec_id], False)

                #fig = plt.figure()
                #fig.add_subplot(1, 2, 1)
                plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0][:,rec_field[1]:(rec_field[3]+1),rec_field[0]:(rec_field[2]+1)]))
                #fig.add_subplot(1, 2, 2)
                #plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0][:,20:60, 40:60]))
                #plt.axis('off')
                plt.show()
                return
            else:
                num_in_cluster_seen = num_in_cluster_seen + 1


# View n images in the cluster_i-th cluster that are closest to the center
def view_nth_cluster(cluster_i, n):
    fig = plt.figure()
    fig_id = 1
    dim_plot = math.floor(math.sqrt(n))
    if dim_plot * dim_plot < n:
        dim_plot = dim_plot + 1

    scores = get_top_n_in_cluster(cluster_i, n)

    for (vec_id, score) in scores:
        print 'Vector #', vec_id, 'with score', score

        fig.add_subplot(dim_plot, dim_plot, fig_id)

        rec_field = rf.get_receptive_field(args.layer, vec_location[vec_id][0], vec_location[vec_id][1])
        load_image(vec_origin_file[vec_id], False)
        plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0][:, rec_field[1]:(rec_field[3]+1), rec_field[0]:(rec_field[2]+1)]))
        plt.axis('off')
        fig_id = fig_id + 1

        if fig_id > n:
            plt.show()
            return

    plt.show()

#### THIS FUNCTION IS TOO SLOW. CAN BE MUCH IMPROVED!!!
def view_n_from_clusters(from_cluster, to_cluster, n_each, save_plots=False):
    fig = plt.figure()
    fig_id = 1

    for i in range(from_cluster, to_cluster+1):
        scores = get_top_n_in_cluster(i, n_each)
        for (vec_id, score) in scores:
            print 'Vector #', vec_id, 'in cluster #', i, ', score:', score
            fig.add_subplot(to_cluster - from_cluster + 1, n_each, fig_id)
            
            rec_field = rf.get_receptive_field(args.layer, vec_location[vec_id][0], vec_location[vec_id][1])
            load_image(vec_origin_file[vec_id], False)
            plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0][:, rec_field[1]:(rec_field[3]+1), rec_field[0]:(rec_field[2]+1)]))

            fig_id = fig_id + 1

            plt.axis('off')
    
    if save_plots:
        plt.savefig(os.path.join(args.save_plots_to, args.layer + '_' + 'clusters' + str(from_cluster) + 'to' + str(to_cluster) + '.png'))
    else:
        plt.show()

if args.save_plots_to is not None:
    matplotlib.use('Agg')
    if not os.path.exists(args.save_plots_to):
        os.makedirs(args.save_plots_to)
    # 8 clusters each time, 16 images for each cluster
    num_rounds = n_clusters / 8
    if n_clusters % 8 != 0:
        num_rounds = num_rounds + 1

    for i in range(num_rounds):
        start = i * 8
        end = start + 7
        if end >= n_clusters:
            end = n_clusters - 1
        view_n_from_clusters(start, end, 16, True)