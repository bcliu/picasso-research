import argparse
import os
from os import walk

# Generate a file list containing images in a given directory, with all the same given label

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_file', required=True)
parser.add_argument('-l', '--label', required=True)
args = parser.parse_args()

in_f = open(args.output_file, 'w')
for (dirpath, dirnames, filenames) in walk(args.input_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        in_f.write(filename + ' ' + args.label + '\n')

in_f.close()
