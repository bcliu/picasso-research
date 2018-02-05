"""File list with labels generator.
Generates a file list containing images in a given directory, with all the same given label.
"""
import argparse
from os import walk
from os import path


def save_file_list(images_dir_path, output_file_path, label):
    with open(output_file_path, 'w') as output_file:
        for (dirpath, dirnames, filenames) in walk(images_dir_path):
            for filename in filenames:
                output_file.write('%s %s\n' % (path.basename(filename), label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path', dest='images_path', required=True)
    parser.add_argument('-o', '--output-file', dest='output_file', required=True)
    parser.add_argument('-l', '--label', required=True)
    args = parser.parse_args()

    save_file_list(args.images_path, args.output_file, args.label)
