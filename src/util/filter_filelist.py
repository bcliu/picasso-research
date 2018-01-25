"""Remove file names in image list with a certain class.
Each line in the input file is in the format of

    image_path image_label

Only images not of the specified label parameter will be saved to the output file.
"""
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input_file', required=True)
parser.add_argument('-o', '--output_file', required=True)
parser.add_argument('-l', '--label', type=int, required=True)
args = parser.parse_args()

in_f = open(args.input_file, 'r')
out_f = open(args.output_file, 'w')

for line in in_f:
    im_name, label = line.split(' ')
    if int(label) != args.label:
        out_f.write(line)
