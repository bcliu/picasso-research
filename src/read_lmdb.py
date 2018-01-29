import argparse

import caffe
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from caffe.proto import caffe_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--lmdb', required=True)
parser.add_argument('--show_one_every', type=int, required=False, default=1)
args = parser.parse_args()

lmdb_file = args.lmdb
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

counter = 0
cycle = 0

for key, value in lmdb_cursor:
    counter += 1
    cycle += 1
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    im = data.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))  # original (dim, col, row)
    print "label ", label, 'shape:', im.shape

    if cycle == args.show_one_every:
        cycle = 0
        plt.imshow(im)
        plt.show()

print counter, ' images stored in lmdb in total'
