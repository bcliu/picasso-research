import caffe

from os import makedirs

from util.parameters_loader import *

caffe.set_device(0)
caffe.set_mode_gpu()


def deprocess_net_image(image):
    """
    Helper function for deprocessing preprocessed images, e.g., for display.
    """
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


def create_caffenet_finetune_fixed_conv3_net(model_name, source_lmdb):
    """
    Creates NetParameter object for finetuning CaffeNet, with conv1, conv2 and conv3 layers fixed,
    using the same parameters as in fixed_conv3_train_val.prototxt
    """
    net_parameter = load_caffenet_net_parameter()

    net_parameter.name = model_name

    layer_names = [l.name for l in net_parameter.layer]

    # Set training data layer parameters
    train_data_layer = net_parameter.layer[0]
    train_data_layer.transform_param.mean_file = path.join(dataset_root, 'ilsvrc12', 'imagenet_mean.binaryproto')
    train_data_layer.data_param.source = source_lmdb
    train_data_layer.data_param.batch_size = 128

    # Set testing data layer parameters
    test_data_layer = net_parameter.layer[1]
    test_data_layer.transform_param.mean_file = train_data_layer.transform_param.mean_file
    test_data_layer.data_param.source = source_lmdb

    # Set learning rate for conv1, conv2 and conv3 to 0 to fix them
    for layer_name in ['conv1', 'conv2', 'conv3']:
        conv_layer_idx = layer_names.index(layer_name)
        conv_layer = net_parameter.layer[conv_layer_idx]
        conv_layer.param[0].lr_mult = 0
        conv_layer.param[1].lr_mult = 0

    accuracy_layer_idx = layer_names.index('accuracy')
    accuracy_layer = net_parameter.layer[accuracy_layer_idx]
    # Add a new accuracy layer for top-5 accuracy
    accuracy_top5_layer = caffe_pb2.LayerParameter()
    accuracy_top5_layer.CopyFrom(accuracy_layer)
    accuracy_top5_layer.name = 'accuracy_top5'
    accuracy_top5_layer.top[0] = 'accuracy_top5'
    accuracy_top5_layer.accuracy_param.top_k = 5

    # Shift existing layers one to the end to insert top5 layer in between
    net_parameter.layer.add()
    for idx in range(accuracy_layer_idx + 1, len(layer_names))[::-1]:
        net_parameter.layer[idx + 1].CopyFrom(net_parameter.layer[idx])

    net_parameter.layer[accuracy_layer_idx + 1].CopyFrom(accuracy_top5_layer)

    return net_parameter


def create_caffenet_finetune_fixed_conv3_solver(model_name, source_lmdb):
    """
    Create SolverParameter object for finetuning CaffeNet
    """
    folder_path = path.join(trained_models_root, 'alexnet_models', model_name)
    if not path.exists(folder_path):
        makedirs(folder_path)

    net_parameter = create_caffenet_finetune_fixed_conv3_net(model_name, source_lmdb)
    net_parameter_file_path = path.join(folder_path, model_name + '_train_val.prototxt')
    with open(net_parameter_file_path, 'w') as net_parameter_file:
        net_parameter_file.write(str(net_parameter))

    solver_parameter = load_caffenet_solver_parameter()

    solver_parameter.net = net_parameter_file_path
    solver_parameter.test_iter[0] = 100
    solver_parameter.base_lr = 0.001
    solver_parameter.stepsize = 20000
    solver_parameter.max_iter = 100000
    solver_parameter.snapshot_prefix = path.join(folder_path, 'snapshot')

    solver_parameter_file_path = path.join(folder_path, model_name + '_solver.prototxt')
    with open(solver_parameter_file_path, 'w') as solver_parameter_file:
        solver_parameter_file.write(str(solver_parameter))

    return solver_parameter_file_path


def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'accuracy', 'accuracy_top5')
    loss, acc, acc_top5 = ({name: np.zeros(niter) for name, _ in solvers}
                           for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it], acc_top5[name][it] = (s.net.blobs[b].data.copy()
                                                                 for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, accuracy_top1=%2d%%, accuracy_top5=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights


solver_param_file = create_caffenet_finetune_fixed_conv3_solver(
    'flickr_40k_conv32_5px_256_with_ilsvrc_fixed_conv3',
    path.join(dataset_root, 'flickr', 'flickr_40k_conv32_5px_256x256_with_ilsvrc_lmdb'))
pretrained_model_path = path.join(original_models_root, 'bvlc_reference_caffenet', 'bvlc_reference_caffenet.caffemodel')
solver = caffe.get_solver(solver_param_file)
solver.net.copy_from(pretrained_model_path)


def run_test():
    niter = 200  # number of iterations to train

    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

    # Delete solvers to save memory.
    del solver, solvers
