"""Remove file names in image list with a certain class.
Each line in the input file is in the format of

    image_path image_label

Only images not of the specified label parameter will be saved to the output file.
"""
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input-file', dest='input_file', required=True)
parser.add_argument('-o', '--output-file', dest='output_file', required=True)
parser.add_argument('-l', '--label', type=int, required=True)
args = parser.parse_args()

with open(args.input_file, 'r') as input_file, open(args.output_file, 'w') as output_file:
    for line in input_file:
        _, label = line.split(' ')
        if int(label) != args.label:
            output_file.write(line)
