from os import path
import tempfile

import google.protobuf.text_format as text_format
from caffe.proto import caffe_pb2
from pylab import *

from env.env import *


def load_net_parameter_file(prototxt_path):
    """
    Loads net parameters stored in a prototxt to a caffe_pb2.NetParameter() object
    """
    net_parameter = caffe_pb2.NetParameter()
    with open(prototxt_path) as prototxt_file:
        prototxt_text = prototxt_file.read()
        text_format.Merge(prototxt_text, net_parameter)
    return net_parameter


def load_solver_parameter_file(prototxt_path):
    """
    Loads solver parameters stored in a prototxt to a caffe_pb2.SolverParameter() object
    """
    solver_parameter = caffe_pb2.SolverParameter()
    with open(prototxt_path) as prototxt_file:
        prototxt_text = prototxt_file.read()
        text_format.Merge(prototxt_text, solver_parameter)
    # We should never use CPU mode
    solver_parameter.solver_mode = caffe_pb2.SolverParameter.GPU
    return solver_parameter


def write_parameter_to_tempfile(parameter):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(parameter))
        return f.name


def load_caffenet_training_net_parameter():
    """
    Loads default CaffeNet NetParameter for training and evaluation
    """
    return load_net_parameter_file(path.join(original_models_root, 'bvlc_reference_caffenet', 'train_val.prototxt'))


def load_caffenet_training_solver_parameter():
    """
    Loads default CaffeNet SolverParameter
    """
    return load_solver_parameter_file(path.join(original_models_root, 'bvlc_reference_caffenet', 'solver.prototxt'))


def load_caffenet_test_net_parameter():
    return load_net_parameter_file(
        path.join(research_root, 'caffe_models', 'bvlc_reference_caffenet', 'test.prototxt'))


def create_test_net(images_lmdb_path, batch_size=50, crop_size=227):
    """
    Creates NetParameter for testing, with test images stored in an lmdb
    Returns a tuple of (net_name, net_path)
    """
    net_parameter = load_caffenet_test_net_parameter()
    net_parameter.name = path.basename(images_lmdb_path) + '_test_net'

    data_layer = net_parameter.layer[0]
    data_layer.transform_param.mean_file = path.join(dataset_root, 'ilsvrc12', 'imagenet_mean.binaryproto')
    data_layer.transform_param.crop_size = crop_size
    data_layer.data_param.source = images_lmdb_path
    data_layer.data_param.batch_size = batch_size

    return net_parameter.name, write_parameter_to_tempfile(net_parameter)
