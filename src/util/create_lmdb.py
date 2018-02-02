"""
Creates lmdb files used for training and testing
"""
import argparse
import subprocess
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images-paths', dest='images_paths', nargs='+', required=True)
parser.add_argument('-l', '--image-label-lists', dest='image_label_lists', nargs='+', required=True)
parser.add_argument('--resize-to-256', dest='should_resize', required=False, default=False, action='store_true',
                    help='Whether images should be resized to 256x256 if they are not already')
args = parser.parse_args()

images_paths = args.images_paths
image_label_lists = args.image_label_lists

if len(images_paths) != len(image_label_lists):
    print '%d image directories provided, but %d label lists are given' % (len(images_paths), len(image_label_lists))
    exit()

resize_parameters = ''
# TODO: DIFFERENCE BETWEEN 227 (input size of alexnet), 224 (input size of vgg) AND 256!!
if args.should_resize:
    print 'Resizing images to 256x256.'
    resize_parameters = '--resize_height=%d --resize_width=%d' % (256, 256)

for images_path, label_list in zip(images_paths, image_label_lists):
    abs_images_path = path.abspath(images_path)
    abs_label_list_path = path.abspath(label_list)
    # Output path is input images directory + _lmdb
    abs_output_path = path.join(path.dirname(abs_images_path), path.basename(abs_images_path) + '_lmdb')

    print 'Creating lmdb from images in %s with labels %s' % (abs_images_path, abs_label_list_path)

    command = 'GLOG_logtostderr=1 convert_imageset %s --shuffle %s %s %s' % (
        resize_parameters, abs_images_path, abs_label_list_path, abs_output_path)

    print '\n' + command + '\n'

    print subprocess.check_output(command.split(' '), stderr=subprocess.STDOUT)

print 'Done.'
