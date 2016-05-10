# Steps:
# 1. Specify image input and output paths
# 2. Specify which neurons activation considered useful for jittering. Specify thresholds if necessary
#           Two options: either rely on clusters, or rely on neurons
# 3. Merge regions, ignore boundaries, to form a region for jittering
# 4. Randomly move regions based on given radius, while avoiding collision
# 5. Smooth out boundaries

# TODO: Try the neuron-based approach as well?
# TODO: another concern: maybe better to adjust ratio to make it square
# TODO: TRY FILL MOVED REGION WITH DOMINANT COLOR OR GRADIENT? OR JUST GAUSSIAN SMOOTH IT?
# TODO: Ignore group if few patches are in it. meaning outlier

from constants import *
import argparse
from os import walk
import os, sys, math, pickle, matplotlib
import numpy as np
import get_receptive_field as rf
import random

# Set Caffe output level to Warnings
os.environ['GLOG_minloglevel'] = '2'

sys.path.insert(0, caffe_root + 'python')
import caffe
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-ld', '--layer_dump', required=True)
parser.add_argument('-cd', '--clusters_dump', required=True)
parser.add_argument('-c', '--clusters', nargs='+', required=True)
parser.add_argument('-r', '--radius', required=True, help='Jitter radius')
parser.add_argument('--interactive', action='store_true', default=False, required=False,
    help='Show which parts are jittered in a screen instead of saving')
args = parser.parse_args()

if not args.interactive:
    matplotlib.use('Agg') # For saving images to file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

net = caffe.Classifier(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
        caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2, 1, 0),
        raw_scale=255,
        image_dims=(224, 224))
net.blobs['data'].reshape(1, 3, 224, 224) #??? necessary?

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
    print '\nDone.'
    
# Important variables:
# 0. vectors: all image patches
# 1. predicted: cluster of each vector
# 2. kmeans_scores: distance of each vector to cluster center

def load_image(path):
    net.predict([caffe.io.load_image(path)], oversample=False)
    

class UnionFind:
    
    obj_arr = []
    group_arr = []
    # A passed-in function which decides if two objects should be merged
    # Returns 0 if yes, 1 otherwise
    compare_fun = None
    
    def union(self, i, j):
        i_root = self.find_root(i)
        j_root = self.find_root(j)
        
        if i_root == j_root:
            return
        self.group_arr[j] = i_root
        
    
    def find_root(self, i):
        if self.group_arr[i] == i:
            return i
        return self.find_root(self.group_arr[i])
    
    
    def merge_all(self):
        for i in range(len(self.obj_arr)):
            for j in range(i + 1, len(self.obj_arr)):
                if self.compare_fun(self.obj_arr[i], self.obj_arr[j]) == 0:
                    self.union(i, j)
    
    def __init__(self, data_arr, compare_fun):
        self.obj_arr = data_arr
        # Initialize group IDs
        self.group_arr = range(len(data_arr))
        self.compare_fun = compare_fun


# Returns 0 if the squares have significant overlaps, 1 otherwise
def are_squares_overlap(tup1, tup2):
    overlap_threshold = 0.25
    
    # Can't be any overlap at all
    left = None
    right = None
    top = None
    bottom = None
    if tup1[0] < tup2[0]:
        left = tup1
        right = tup2
    else:
        left = tup2
        right = tup1
    if tup1[1] < tup2[1]:
        top = tup1
        bottom = tup2
    else:
        top = tup2
        bottom = tup1
    if left[2] < right[0] or top[3] < bottom[1]:
        return 1
        
    p1_x = max(tup1[0], tup2[0])
    p1_y = max(tup1[1], tup2[1])
    p2_x = min(tup1[2], tup2[2])
    p2_y = min(tup1[3], tup2[3])
    
    area_overlap = (p2_x - p1_x) * (p2_y - p1_y) * 1.0
    area1 = (tup1[2] - tup1[0]) * (tup1[3] - tup1[1]) * 1.0
    area2 = (tup2[2] - tup2[0]) * (tup2[3] - tup2[1]) * 1.0
    
    if area_overlap / area1 > overlap_threshold or area_overlap / area2 > overlap_threshold:
        return 0
    return 1
    
    
# Given an array of squares (four-tuples, of coordinates of topleft and bottomright pixels),
# Merge those with large overlapping areas into larger shapes
def merge_squares(squares):
    uf = UnionFind(squares, are_squares_overlap)
    uf.merge_all()
    
    #print 'Grouping results:', uf.group_arr
    
    merged_squares = {}
    
    # Taking the simple approach: just generate a rectangle that contains all those squares
    for i in range(len(squares)):
        square = squares[i]
        group = uf.group_arr[i]
        
        if group not in merged_squares:
            merged_squares[group] = [float('inf'), float('inf'), -1, -1]
        
        merged_squares[group][0] = min(merged_squares[group][0], square[0])
        merged_squares[group][1] = min(merged_squares[group][1], square[1])
        merged_squares[group][2] = max(merged_squares[group][2], square[2])
        merged_squares[group][3] = max(merged_squares[group][3], square[3])
    
    return merged_squares.values()
    

def jitter_regions(im, regions, radius):
    gradient_percentage = 0.5
    jittered = im.copy()
    for region in regions:
        patch = im[region[1]:(region[3]+1), region[0]:(region[2]+1), :]
        region_height = int(region[3] - region[1])
        region_width = int(region[2] - region[0])
        
        jitter_y = int(random.randrange(-radius, radius))
        jitter_x = int(math.floor(math.sqrt(radius ** 2 - jitter_y ** 2) * np.random.choice([-1, 1])))
        
        new_y1 = region[1] + jitter_y
        # Check boundaries
        if new_y1 < 0:
            new_y1 = 0
        if region_height + new_y1 >= len(im):
            new_y1 = len(im) - region_height - 1
        new_y2 = region_height + new_y1
        
        new_x1 = region[0] + jitter_x
        if new_x1 < 0:
            new_x1 = 0
        if region_width + new_x1 >= len(im[0]):
            new_x1 = len(im[0]) - region_width - 1
        new_x2 = region_width + new_x1
        
        assert(new_x1 >= 0 and new_x2 >= 0 and new_y1 >= 0 and new_y2 >= 0)
        assert(new_x1 < len(im[0]) and new_x2 < len(im[0]) and new_y1 < len(im) and new_y2 < len(im))
        assert(new_x2 - new_x1 == region[2] - region[0])
        assert(new_y2 - new_y1 == region[3] - region[1])
        
        center_x = region_width / 2 + new_x1
        center_y = region_height / 2 + new_y1
        farthest = math.sqrt((center_x - new_x1) ** 2 + (center_y - new_y1) ** 2)
        for y in range(new_y1, new_y2+1):
            for x in range(new_x1, new_x2+1):
                # Distance from center of patch
                dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                patch_opacity = (1 - dist / farthest) / gradient_percentage # TODO CHANGE THE OPACITY FUNCTION for a smoother and faster
                if patch_opacity > 1:
                    patch_opacity = 1
                jittered[y, x, :] = (1 - patch_opacity) * im[y, x, :] + patch_opacity * patch[y-new_y1, x-new_x1, :]
    
    return jittered

clusters = []
for c in args.clusters:
    clusters.append(int(c))

for (dirpath, dirnames, filenames) in walk(args.input_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        print 'Processed', path
        name_only, ext = os.path.splitext(filename)
        
        load_image(path)
        
        dim_feature_map = len(net.blobs[layer].data[0][0])
        im = net.transformer.deprocess('data', net.blobs['data'].data[0])
        plt.imshow(im)
        axis = plt.gca()
        
        # Maps cluster ID to kmeans score threshold. Patches with activation larger than threshold
        # (meaning distance too large) will not be considered
        thresholds = {}
        thres_percentage = 0.8
        for c in clusters:
            cluster_scores = [kmeans_scores[i] for i in range(n_vectors) if predicted[i] == c]
            cluster_scores.sort(reverse=True)
            thresholds[c] = cluster_scores[int(math.floor(len(cluster_scores) * thres_percentage)) - 1]
        
        detected_squares = []
        
        for y in range(dim_feature_map):
            for x in range(dim_feature_map):
                hypercolumn = net.blobs[layer].data[0][:,y,x].copy().reshape(1, -1)
                prediction = kmeans_obj.predict(hypercolumn)
                if prediction in clusters:
                    rec_field = rf.get_receptive_field(layer, x, y)
                    axis.add_patch(Rectangle((rec_field[0], rec_field[1]),
                        rec_field[2] - rec_field[0] + 1,
                        rec_field[3] - rec_field[1] + 1,
                        fill=False, edgecolor="red"))
                    detected_squares.append(rec_field)
                    
        merged_regions = merge_squares(detected_squares)
        for region in merged_regions:
            axis.add_patch(Rectangle((region[0], region[1]),
                region[2] - region[0] + 1,
                region[3] - region[1] + 1,
                fill=False, edgecolor="blue"))
        
        #plt.show()
        if not args.interactive:
            plt.savefig(os.path.join(args.output_dir, name_only + '_detected' + ext))
        
        plt.clf()
        
        jittered = jitter_regions(im, merged_regions, int(args.radius))
        plt.imshow(jittered)
        if args.interactive:
            plt.show()
        else:
            plt.savefig(os.path.join(args.output_dir, name_only + '_jit' + ext))