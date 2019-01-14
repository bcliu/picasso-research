"""
Creates lmdb files used for training and testing
"""
import subprocess
import re

from generate_filelist import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images-paths', dest='images_paths', nargs='+', required=True)
parser.add_argument('-g', '--generate-label-lists-with-label', dest='gen_list_with_label', required=False,
                    help='Generate a label list with all images in the input path, with the same provided label')
parser.add_argument('-l', '--image-label-lists', dest='image_label_lists', nargs='+', required=False,
                    help=['If not using auto generated label lists, provide label list files after this parameter. '
                          'The number of list files should be equal to that of images paths.'])
parser.add_argument('--resize-to-256', dest='should_resize', required=False, default=False, action='store_true',
                    help='Whether images should be resized to 256x256 if they are not already')
args = parser.parse_args()

images_paths = args.images_paths
image_label_lists = args.image_label_lists

if args.gen_list_with_label is not None:
    image_label_lists = []
    # Generate temporary list files with the provided label
    for images_path in images_paths:
        abs_images_path = path.abspath(images_path)
        label_list_path = path.join(path.dirname(abs_images_path),
                                    '%s_with_label_%s.txt' % (path.basename(abs_images_path), args.gen_list_with_label))

        if path.exists(label_list_path):
            print '%s already exists. Move or rename this file first.' % label_list_path
            exit()

            save_single_label_images_list(images_path, label_list_path, args.gen_list_with_label)
        image_label_lists.append(label_list_path)

        print 'Created file list with label %s in %s' % (args.gen_list_with_label, label_list_path)

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

    if path.exists(abs_output_path):
        print '%s already exists -- rename or move the existing path' % abs_output_path
        exit()

    print 'Creating lmdb from images in %s with labels %s' % (abs_images_path, abs_label_list_path)

    command = 'convert_imageset %s --shuffle %s/ %s %s' % (
        resize_parameters, abs_images_path, abs_label_list_path, abs_output_path)

    print '\n' + command + '\n'

    p = subprocess.Popen(re.split('\s+', command), stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
    for line in p.stdout:
        print line
    p.stdout.close()

print 'Done.'
