import argparse
import subprocess
from os import walk
import os


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--caffe_models', nargs='+', required=True)
parser.add_argument('-t', '--tests_dir', required=True)
parser.add_argument('--gpu', required=False, default='1')
args = parser.parse_args()

test_paths = []

for (dirpath, dirnames, filenames) in walk(args.tests_dir):
    for filename in filenames:
        if filename.endswith('.prototxt'):
            test_paths.append(os.path.abspath(os.path.join(dirpath, filename)))

test_paths.sort()

for model in args.caffe_models:
    caffe_model = os.path.abspath(model)
    print Colors.WARNING + 'Testing caffe model ' + os.path.basename(caffe_model) + Colors.ENDC

    for test in test_paths:
        print Colors.WARNING + 'on ' + os.path.basename(test) + ':' + Colors.ENDC
        command = "caffe test -weights " + caffe_model + " -gpu " + args.gpu + " -model " + test
        print Colors.OKBLUE + 'command: ' + command + Colors.ENDC
        out = subprocess.check_output(command.split(' '), stderr=subprocess.STDOUT).splitlines()
        print out[-3]
        print out[-2]

print 'Done.'

# MAKE IT SO THAT THE PROTOTXT ARE GENERATED AUTOMATICALLY FROM TEMPLATE!!