"""Grayscale image converter.
Converts all images in a given input directory to grayscale, and save them to an output directory.
"""
import argparse
import os
from os import walk

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for (dirpath, dirnames, filenames) in walk(args.input_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        im = Image.open(path).convert('L')
        out_path = os.path.abspath(os.path.join(args.output_dir, filename))
        im.save(out_path)
