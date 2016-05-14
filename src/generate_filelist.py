import argparse
import os
from os import walk

parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--train_dir', required=True)
parser.add_argument('-v', '--val_dir', required=True)
parser.add_argument('-l', '--label', required=True)
args = parser.parse_args()

train_f = open('train.txt', 'w')
val_f = open('val.txt', 'w')
for (dirpath, dirnames, filenames) in walk(args.train_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        train_f.write(filename + ' ' + args.label + '\n')

for (dirpath, dirnames, filenames) in walk(args.val_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        val_f.write(filename + ' ' + args.label + '\n')

train_f.close()
val_f.close()
