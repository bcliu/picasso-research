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


def load_caffenet_net_parameter():
    """
    Loads default CaffeNet NetParameter for training and evaluation
    """
    return load_net_parameter_file(original_models_root + 'bvlc_reference_caffenet/train_val.prototxt')


def load_caffenet_solver_parameter():
    """
    Loads default CaffeNet SolverParameter
    """
    return load_solver_parameter_file(original_models_root + 'bvlc_reference_caffenet/solver.prototxt')