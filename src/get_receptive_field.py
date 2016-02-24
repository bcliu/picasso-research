from constants import *

# Layers and parameters of VGG16
layers = [
    'data',
    'conv1_1',
    'conv1_2',
    'pool1',
    'conv2_1',
    'conv2_2',
    'pool2',
    'conv3_1',
    'conv3_2',
    'conv3_3',
    'pool3',
    'conv4_1',
    'conv4_2',
    'conv4_3',
    'pool4'
]

# [ pad, kernel_size ]
conv_params = {
    'conv1_1': [1, 3],
    'conv1_2': [1, 3],
    'conv2_1': [1, 3],
    'conv2_2': [1, 3],
    'conv3_1': [1, 3],
    'conv3_2': [1, 3],
    'conv3_3': [1, 3],
    'conv4_1': [1, 3],
    'conv4_2': [1, 3],
    'conv4_3': [1, 3]
}

# [stride, kernel_size]
pool_params = {
    'pool1': [2, 2],
    'pool2': [2, 2],
    'pool3': [2, 2],
    'pool4': [2, 2]
}

layer_dims = {
    'data': 224,
    'conv1_1': 224,
    'conv1_2': 224,
    'pool1': 112,
    'conv2_1': 112,
    'conv2_2': 112,
    'pool2': 56,
    'conv3_1': 56,
    'conv3_2': 56,
    'conv3_3': 56,
    'pool3': 28
}

prev_layers = {}
# Build dictionary that maps one layer to its prev layer
for i in range(len(layers) - 1):
    prev_layers[layers[i+1]] = layers[i]


# Returns: top-left corner x, y, and bottom-right corner x, y
# x is the column of the neuron in the 2D array representing the filter response
# y is the row
# NOTE: THERE COULD BE PROBLEM HERE. DOUBLE CHECK
def get_conv_neuron_rec_field(layer, neuron_x, neuron_y):
    last_layer_top_left = [-1, -1]
    last_layer_bottom_right = [-1, -1]

    # stride == 1 for VGG16
    params = conv_params[layer]
    pad = params[0]
    kernel_size = params[1]

    last_layer_top_left[0] = neuron_x - pad
    last_layer_top_left[1] = neuron_y - pad
    last_layer_bottom_right[0] = neuron_x - pad + kernel_size - 1
    last_layer_bottom_right[1] = neuron_y - pad + kernel_size - 1

    last_layer = prev_layers[layer]

    if last_layer.startswith('conv'):
        last_layer_tl_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                            last_layer_top_left[0],
                                                            last_layer_top_left[1])
        last_layer_br_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                            last_layer_bottom_right[0],
                                                            last_layer_bottom_right[1])
    elif last_layer.startswith('pool'):
        last_layer_tl_rec_field = get_pool_neuron_rec_field(prev_layers[layer],
                                                            last_layer_top_left[0],
                                                            last_layer_top_left[1])
        last_layer_br_rec_field = get_pool_neuron_rec_field(prev_layers[layer],
                                                            last_layer_bottom_right[0],
                                                            last_layer_bottom_right[1])
    else:
        # Data layer
        return [last_layer_top_left[0], last_layer_top_left[1],
                last_layer_bottom_right[0], last_layer_bottom_right[1]]

    return [last_layer_tl_rec_field[0], last_layer_tl_rec_field[1],
            last_layer_br_rec_field[2], last_layer_br_rec_field[3]]


# Returns: same as the above
def get_pool_neuron_rec_field(layer, neuron_x, neuron_y):
    # [x, y] of top-left neuron in the previous layer this neuron is looking at
    last_layer_top_left = [-1, -1]
    last_layer_bottom_right = [-1, -1]

    params = pool_params[layer]
    stride = params[0]
    kernel_size = params[1]

    last_layer_top_left[0] = stride * neuron_x
    last_layer_top_left[1] = stride * neuron_y
    last_layer_bottom_right[0] = stride * neuron_x + kernel_size - 1
    last_layer_bottom_right[1] = stride * neuron_y + kernel_size - 1

    last_layer_tl_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                        last_layer_top_left[0],
                                                        last_layer_top_left[1])
    last_layer_br_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                        last_layer_bottom_right[0],
                                                        last_layer_bottom_right[1])

    return [last_layer_tl_rec_field[0], last_layer_tl_rec_field[1],
            last_layer_br_rec_field[2], last_layer_br_rec_field[3]]


def get_receptive_field(layer, x, y):
    rec_field = None
    if layer.startswith('pool'):
        rec_field = get_pool_neuron_rec_field(layer, x, y)
    elif layer.startswith('conv'):
        rec_field = get_conv_neuron_rec_field(layer, x, y)

    return rec_field