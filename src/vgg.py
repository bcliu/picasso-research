from constants import *
import argparse
from os import walk
import os

import matplotlib

parser = argparse.ArgumentParser(description='')
parser.add_argument('--images', default=research_root + 'images/flickr/eyes-yes/', required=False)
parser.add_argument('--layer', default='conv4_1', required=False)
parser.add_argument('--sample_fraction', default=0.3, required=False)
parser.add_argument('--n_clusters', default=32, required=False)

parser.add_argument('--center_only_path', default=None, required=False)
parser.add_argument('--center_only_neuron_x', default=None, required=False)

parser.add_argument('--gpu', default=0, required=False)

parser.add_argument('--load_layer_dump_from', default=None, required=False)
parser.add_argument('--load_classification_dump_from', default=None, required=False)

parser.add_argument('--save_layer_dump_to', default=None, required=False)
parser.add_argument('--save_classification_dump_to', default=None, required=False)

parser.add_argument('--save_plots_to', default=None, required=False)

args = parser.parse_args()

if args.save_plots_to is not None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import math
import numpy as np
import pickle

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

import get_receptive_field as rf

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

# Force non-interative mode, if saving plots
if args.save_plots_to is not None:
    plt.ioff()

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

if args.load_layer_dump_from is not None:
    print 'Loading raw layer dump file from', args.load_layer_dump_from
    f = open(args.load_layer_dump_from)
    [args_images, args_layer, args_sample_fraction,
        args_center_path, args_center_x,
        sample_mask, vectors, vec_origin_file, vec_location] = pickle.load(f)
    
    args.images = args_images
    args.layer = args_layer
    args.sample_fraction = args_sample_fraction
    args.center_only_path = args_center_path
    args.center_only_neuron_x = args_center_x

    n_clusters = int(args.n_clusters)
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

    # Save data (layer) dump if parameter is specified
    if args.save_layer_dump_to is not None:
        print 'Saving layer data dump to', args.save_layer_dump_to
        f = open(args.save_layer_dump_to, 'wb')
        pickle.dump([args.images, args.layer, args.sample_fraction,
            args.center_only_path, args.center_only_neuron_x,
            sample_mask, vectors, vec_origin_file, vec_location], f)
        f.close()
        print 'Finished saving layer dump'

print 'Got', len(vectors), 'vectors in total for clustering'

# TRY PCA, ICA AS WELL!!!
# if this method works, continue to explore how to make your initial method work as well

if args.load_classification_dump_from is not None:
    print 'Loading classification dump from', args.load_classification_dump_from
    f = open(args.load_classification_dump_from)
    n_clusters, kmeans_obj, predicted = pickle.load(f)
    args.n_clusters = n_clusters
    n_clusters = int(n_clusters)
    print 'Finished loading classification dump'
else:
    n_clusters = int(args.n_clusters)
    n_restarts = 10
    kmeans_obj = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_restarts)
    predicted = kmeans_obj.fit_predict(vectors)

    if args.save_classification_dump_to is not None:
        print 'Saving classification dump to', args.save_classification_dump_to
        f = open(args.save_classification_dump_to, 'wb')
        pickle.dump([n_clusters, kmeans_obj, predicted], f)
        f.close()
        print 'Finished saving classification dump'
        
def do_tsne(data):
    tsne_model = TSNE(n_components=2, init='pca')
    trans_tsne = tsne_model.fit_transform(data)
    return tsne_model, trans_tsne

## WARNING: THIS METHOD OF SELECTING MAY NOT BE WORKING AS YOU INTENDED!!!
def plot_clusters_2d(data, labels, selected_rows):
    plt.scatter(data[selected_rows, 0], data[selected_rows, 1], c=labels[selected_rows], cmap=plt.get_cmap('Spectral'), lw=0)
    plt.show()

#plot_clusters_2d(vectors20k_tsne[np.logical_or(predicted20k == 26, predicted20k == 54), :], predicted20k[np.logical_or(predicted20k == 26, predicted20k == 54)])

##### PCA
pca = PCA(n_components=80)
#vectors_trans = pca.fit_transform(vectors)
#print pca.explained_variance_ratio_
#predicted = kmeans_obj.fit_predict(vectors_trans)


#########

def get_sparsity(data):
    # From http://www.cnbc.cmu.edu/~samondjm/papers/VinjeandGallant2000.pdf
    # ??? BUT THIS ONLY MEASURES NEURON ON STIMULI? NOT REVERSE?
    numer = 0
    denom = 0
    n = len(data) * 1.0
    for r in data:
        numer = numer + r / n
        denom = denom + r**2 / n
        
    A = numer**2 / denom
    S = (1 - A) / (1 - 1/n)
    return A, S

def get_activations_of_cluster(cluster_i):
    count = 0
    bigsum = None
    for idx in range(0, len(predicted)):
        if predicted[idx] == cluster_i:
            if bigsum is None:
                bigsum = vectors[idx]
            else:
                bigsum = bigsum + vectors[idx]
            count = count + 1.0

    return bigsum, count

def plot_raw_activation(cluster_i):
    totalsum, count = get_activations_of_cluster(cluster_i)
    totalsum = totalsum / count

    plt.plot(range(0, len(totalsum)), totalsum, 'b-')
    plt.title(args.layer + ' Cluster #' + str(cluster_i) + ' (' + str(args.n_clusters) + ' total)')
    plt.xlabel('Neuron #')
    plt.ylabel('Average activation')

    A, S = get_sparsity(totalsum)
    plt.annotate(
        'Sparsity: S=' + str(S),
        xy = (0.9, 0.9), xytext = (0.9, 0.9),
        textcoords = 'axes fraction', ha = 'right', va = 'bottom')

    plt.show()

def plot_activation(cluster_i, top_n=4):
    bigsum, count = get_activations_of_cluster(cluster_i)
    bigsum = bigsum / count
    
    # Get sorted indexes
    sorted_indexes = [i[0] for i in sorted(enumerate(bigsum), key=lambda x:x[1], reverse=True)]
    top_indexes = [sorted_indexes[x] for x in range(top_n)]
    top_responses = [bigsum[sorted_indexes[x]] for x in range(top_n)]

    bigsum.sort()
    bigsum = bigsum[::-1]
    plt.plot(range(len(bigsum)), bigsum, 'bo-')
    plt.title(args.layer + ' Cluster #' + str(cluster_i) + ' (' + str(args.n_clusters) + ' total)')
    plt.xlabel('Neuron #')
    plt.ylabel('Average activation')

    print 'Highest neuron responses:'
    for i in range(top_n):
        print 'Neuron #', top_indexes[i], ', mean response: ', top_responses[i]

        plt.annotate(
            'Neuron #' + str(top_indexes[i]) + '\navg: ' + str(top_responses[i]),
            xy = (i, bigsum[i]), xytext = (100 + 20 * i, -50 - 20 * i),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    # Print sparsity metric
    A, S = get_sparsity(bigsum)
    plt.annotate(
        'Sparsity: S=' + str(S),
        xy = (0.9, 0.9), xytext = (0.9, 0.9),
        textcoords = 'axes fraction', ha = 'right', va = 'bottom')
    plt.show()

# Plot response of neuron_i on all patches in vectors array
def plot_stimuli_response(neuron_i, inputs, top_n=16):
    fig = plt.figure()

    all_vectors = np.array(inputs)
    responses = all_vectors[:, neuron_i]
    len_responses = len(responses)

    sorted_indexes = [i[0] for i in sorted(enumerate(responses), key=lambda x:x[1], reverse=True)]
    responses.sort()
    responses = responses[::-1]
    plt.plot(range(len(all_vectors)), responses, 'bo-')
    plt.title(args.layer + ' Neuron #' + str(neuron_i) + ' Responses to ' + str(len_responses) + ' Stimuli')
    plt.xlabel('Stimulus #')
    plt.ylabel('Activation')

    # Include patches with top 8 responses
    for i in range(top_n):
        this_vec_location = vec_location[sorted_indexes[i]]
        rec_field = rf.get_receptive_field(args.layer, this_vec_location[0], this_vec_location[1])
        load_image(vec_origin_file[sorted_indexes[i]], False)
        im = net.transformer.deprocess('data',
            net.blobs['data'].data[0][:,rec_field[1]:(rec_field[3]+1),rec_field[0]:(rec_field[2]+1)])
        fig.figimage(im, 100 + i * 50, 20)

    # Find number of stimuli that generate response > 0.5*MAX
    num_half_height_stimuli = 0
    for i in range(len_responses):
        if responses[i] > 0.5 * responses[0]:
            num_half_height_stimuli = num_half_height_stimuli + 1

    print 'Percentage of responses greater than half height:', num_half_height_stimuli * 1.0 / len_responses

    plt.show()

## Plot image with response of a certain neuron, or highest among all neurons in each hypercolumn
def dye_image_with_response(path):
    pass

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

print 'All tasks completed. Exiting'
