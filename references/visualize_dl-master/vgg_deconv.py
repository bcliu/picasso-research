from constants import *

import os, sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imsave
from visualize_cls_salience import visualize
from deconv_net import deconv_net

input_im = sys.argv[1]

# load net model
caffe.set_mode_gpu()
net = caffe.Net('./data/vgg16_mod.prototxt', caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)

# show net details
print '\n blobs of caffe  model:'
for k,value in net.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

print '\n params of caffe model:'
for k,value in net.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)


# set deconv_net
deconv5_3 = caffe.Net('./data/vgg/deconv5_3.prototxt',caffe.TEST)
deconv5_2 = caffe.Net('./data/vgg/deconv5_2.prototxt',caffe.TEST)
deconv5_1 = caffe.Net('./data/vgg/deconv5_1.prototxt',caffe.TEST)
deconv4_3 = caffe.Net('./data/vgg/deconv4_3.prototxt',caffe.TEST)
deconv4_2 = caffe.Net('./data/vgg/deconv4_2.prototxt',caffe.TEST)
deconv4_1 = caffe.Net('./data/vgg/deconv4_1.prototxt',caffe.TEST)
deconv3_3 = caffe.Net('./data/vgg/deconv3_3.prototxt',caffe.TEST)
deconv3_2 = caffe.Net('./data/vgg/deconv3_2.prototxt',caffe.TEST)
deconv3_1 = caffe.Net('./data/vgg/deconv3_1.prototxt',caffe.TEST)
deconv2_2 = caffe.Net('./data/vgg/deconv2_2.prototxt',caffe.TEST)
deconv2_1 = caffe.Net('./data/vgg/deconv2_1.prototxt',caffe.TEST)
deconv1_2 = caffe.Net('./data/vgg/deconv1_2.prototxt',caffe.TEST)
deconv1_1 = caffe.Net('./data/vgg/deconv1_1.prototxt',caffe.TEST)

deconv5_3.params['deconv'][0].data[...] = net.params['conv5_3'][0].data
deconv5_2.params['deconv'][0].data[...] = net.params['conv5_2'][0].data
deconv5_1.params['deconv'][0].data[...] = net.params['conv5_1'][0].data
deconv4_3.params['deconv'][0].data[...] = net.params['conv4_3'][0].data
deconv4_2.params['deconv'][0].data[...] = net.params['conv4_2'][0].data
deconv4_1.params['deconv'][0].data[...] = net.params['conv4_1'][0].data
deconv3_3.params['deconv'][0].data[...] = net.params['conv3_3'][0].data
deconv3_2.params['deconv'][0].data[...] = net.params['conv3_2'][0].data
deconv3_1.params['deconv'][0].data[...] = net.params['conv3_1'][0].data
deconv2_2.params['deconv'][0].data[...] = net.params['conv2_2'][0].data
deconv2_1.params['deconv'][0].data[...] = net.params['conv2_1'][0].data
deconv1_2.params['deconv'][0].data[...] = net.params['conv1_2'][0].data
deconv1_1.params['deconv'][0].data[...] = net.params['conv1_1'][0].data

print '\n blobs of caffe model:'
for k,value in deconv5_3.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv5_2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv5_1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv4_3.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv4_2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv4_1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv3_3.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv3_2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv3_1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv2_2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv2_1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv1_2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)
for k,value in deconv1_1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)


print '\n params of caffe model:'
for k,value in deconv5_3.params.items():
    print '{}:{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv5_2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv5_1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv4_3.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv4_2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv4_1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv3_3.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv3_2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv3_1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv2_2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv2_1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv1_2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)
for k,value in deconv1_1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)

deconv = deconv_net()
deconv.set_unpool_layer(net.blobs['pool5_mask'].data,2,3,'pool5')
deconv.set_relu_layer('relu5_3')
deconv.set_deconv_layer(deconv5_3,'conv5_3')
deconv.set_relu_layer('relu5_2')
deconv.set_deconv_layer(deconv5_2,'conv5_2')
deconv.set_relu_layer('relu5_1')
deconv.set_deconv_layer(deconv5_1,'conv5_1')
deconv.set_unpool_layer(net.blobs['pool4_mask'].data,2,3,'pool4')
deconv.set_relu_layer('relu4_3')
deconv.set_deconv_layer(deconv4_3,'conv4_3')
deconv.set_relu_layer('relu4_2')
deconv.set_deconv_layer(deconv4_2,'conv4_2')
deconv.set_relu_layer('relu4_1')
deconv.set_deconv_layer(deconv4_1,'conv4_1')
deconv.set_unpool_layer(net.blobs['pool3_mask'].data,2,3,'pool3')
deconv.set_relu_layer('relu3_3')
deconv.set_deconv_layer(deconv3_3,'conv3_3')
deconv.set_relu_layer('relu3_2')
deconv.set_deconv_layer(deconv3_2,'conv3_2')
deconv.set_relu_layer('relu3_1')
deconv.set_deconv_layer(deconv3_1,'conv3_1')
deconv.set_unpool_layer(net.blobs['pool2_mask'].data,2,3,'pool2')
deconv.set_relu_layer('relu2_2')
deconv.set_deconv_layer(deconv2_2,'conv2_2')
deconv.set_relu_layer('relu2_1')
deconv.set_deconv_layer(deconv2_1,'conv2_1')
deconv.set_unpool_layer(net.blobs['pool1_mask'].data,2,3,'pool1')
deconv.set_relu_layer('relu1_2')
deconv.set_deconv_layer(deconv1_2,'conv1_2')
deconv.set_relu_layer('relu1_1')
deconv.set_deconv_layer(deconv1_1,'conv1_1')

# read image
im = caffe.io.load_image(input_im)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_mean('data',np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_channel_swap('data',[2,1,0])
transformer.set_raw_scale('data',224)
transformer.set_transpose('data',[2,0,1])

#transform image to net input
input_img = transformer.preprocess('data',im)
input_img = input_img.reshape([1] + [input_img.shape[i] for i in range(3)])

#get feature
out = net.forward(data=input_img)

# set switch of pool
deconv.set_unpool_layer(net.blobs['pool5_mask'].data,2,2,'pool5')
deconv.set_unpool_layer(net.blobs['pool4_mask'].data,2,2,'pool4')
deconv.set_unpool_layer(net.blobs['pool3_mask'].data,2,2,'pool3')
deconv.set_unpool_layer(net.blobs['pool2_mask'].data,2,2,'pool2')
deconv.set_unpool_layer(net.blobs['pool1_mask'].data,2,2,'pool1')

# find top activation and reconstruction
layer = 'conv5_3'
begin_with = 'relu5_3'
top_act = np.zeros(net.blobs[layer].data.shape)
layer_feat_map = net.blobs[layer].data

top_act[layer_feat_map==layer_feat_map.max()] = layer_feat_map.max()
# SO ONLY ACTIVATING ONE NEURON IN ONE HYPERCOLUMN??? WHY THAT MAKES SENSE??
# what happens if you activate all? everything? maybe one in every hypercolumn, or everyone in one hypercolumn?
# and what if you activate same neuron across all hypercolumns..?
recon_feat = deconv.recon_down(top_act, begin_with)

# show reconstruction image
image = recon_feat[-1][0]
image -= image.min()
image /= image.max()
image = image.transpose([1,2,0])
re_image_name = 'vgg_deconv_out.png'
imsave(re_image_name,image)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(transformer.deprocess('data',input_img[0]))
plt.show()
