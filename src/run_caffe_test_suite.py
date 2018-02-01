import argparse
import subprocess

from util.parameters_loader import *
from util.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--caffe_models', nargs='+', required=True)
args = parser.parse_args()

# List of (model_name, model_path) tuples
test_models = []

'''
missing:
flickr/flickr_40k_lmdb
flickr/flickr_test_lmdb
ilsvrc/ilsvrc_val_train_lmdb
ilsvrc/ilsvrc_val_test_lmdb
'''
test_set_dir_names = [
    'cubic_face_lmdb',
    'cubic_grayscale_lmdb',

    'labeled_faces_in_wild/lfw_train_lmdb',
    'labeled_faces_in_wild/lfw_test_lmdb',
    'labeled_faces_in_wild/lfw_train_conv32_5px_lmdb',
    'labeled_faces_in_wild/lfw_train_conv33_15px_lmdb'
]

for test_set_dir in test_set_dir_names:
    test_models.append(create_test_net(path.join(dataset_root, test_set_dir)))

for model in args.caffe_models:
    caffe_model = path.abspath(model)
    print Colors.WARNING + 'Testing caffe model ' + path.basename(caffe_model) + Colors.ENDC

    for model_name, model_path in test_models:
        print Colors.WARNING + 'on ' + model_name + ':' + Colors.ENDC
        command = "caffe test -weights " + caffe_model + " -gpu 0 -model " + model_path
        print Colors.OKBLUE + 'command: ' + command + Colors.ENDC
        out = subprocess.check_output(command.split(' '), stderr=subprocess.STDOUT).splitlines()
        print out[-3]
        print out[-2]

print 'Done.'
