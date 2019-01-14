"""File list with labels generator.
Generates a file list containing images in a given directory, with all the same given label.
"""
import argparse
from os import walk
from os import path
import re

# TODO: write unit tests


def save_single_label_images_list(images_dir_path, output_file_path, label):
    """
    :param images_dir_path: Path to input image directory
    :param output_file_path: Output file list path
    :param label: Label of the images, of integer type
    """
    with open(output_file_path, 'w') as output_file:
        images_list = generate_single_label_images_list(images_dir_path, label, prefix_images_dir_name=False)
        for image_line in images_list:
            output_file.write('%s\n' % image_line)


def generate_single_label_images_list(images_dir_path, label, prefix_images_dir_name=False):
    """
    :param images_dir_path: Path to input image directory
    :param label: Label of the images, of integer type
    :param prefix_images_dir_name: Whether output image path should be prefixed by root directory name.
                                   This is useful when this images_dir is a subdirectory of a dataset directory
                                   containing another subdirectory, such as LFW or ILSVRC
    """
    output_images_list = []
    images_dir_name = path.basename(path.abspath(images_dir_path))

    # Loops through all files recursively through all subdirectories
    for (dirpath, dirnames, filenames) in walk(images_dir_path, followlinks=True):
        for filename in filenames:
            # Generate relative path from the given root directory
            relative_path = path.relpath(path.join(dirpath, filename), images_dir_path)

            if prefix_images_dir_name:
                relative_path = path.join(images_dir_name, relative_path)

            output_images_list.append('%s %d' % (relative_path, label))

    return output_images_list


def generate_mixed_images_list(images_path, labels_path, exclude_label):
    """
    :param images_path:
    :param labels_path:
    :param exclude_label:
    :return:
    """
    output_images_list = []
    abs_images_path = path.abspath(images_path)
    images_dir_name = path.basename(abs_images_path)

    with open(labels_path, 'r') as labels_file:
        for line in labels_file:
            line = line.strip()
            if len(line) == 0:
                continue

            image_name, label = line.split(' ')
            if int(label) != exclude_label:
                new_image_name = path.join(images_dir_name, image_name)
                output_images_list.append('%s %s' % (new_image_name, label))

    return output_images_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--single-label-images-path', dest='single_label_images_path', required=True,
                        help='Path to images in which all have the same label')
    parser.add_argument('-sl', '--label', required=True, help='Label of the images in the first paramter')
    parser.add_argument('-m', '--mixed-images-path', dest='mixed_images_path', required=False,
                        help='Path to images of mixed labels, such as LFW or ILSVRC')
    parser.add_argument('-ml', '--mixed-images-labels-path', dest='mixed_images_labels_path', required=False,
                        help='Path to label list of the images of mixed labels, such as LFW and ILSVRC')
    parser.add_argument('-o', '--output-file', dest='output_file', required=True)
    args = parser.parse_args()

    if args.mixed_images_path is None:
        # Generate just single label images
        save_single_label_images_list(args.single_label_images_path, args.output_file, args.label)
    else:
        # Generate single label images list with directory name prefix, plus mixed label images list, excluding
        # images with the given single label
        single_label_images_list = generate_single_label_images_list(args.single_label_images_path, int(args.label),
                                                                     True)
        mixed_label_images_list = generate_mixed_images_list(args.mixed_images_path, args.mixed_images_labels_path,
                                                             int(args.label))

        with open(args.output_file, 'w') as output_file:
            for image_line in single_label_images_list:
                output_file.write('%s\n' % image_line)
            for image_line in mixed_label_images_list:
                output_file.write('%s\n' % image_line)
