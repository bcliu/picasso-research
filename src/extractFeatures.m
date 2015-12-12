constants;

imagePath = caffe_root + 'examples/images/fish-bike.jpg';

caffe.set_mode_cpu();
model_dir = caffe_root + 'models/bvlc_alexnet/';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'bvlc_alexnet.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
net = caffe.Net(net_model, net_weights, phase);
im = imread(imagePath);

% prepare oversampled input
% input_data is Height x Width x Channel x Num
prepared = prepare_image(im);
input_data = { prepared };

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);

scores = scores{1};
scores = mean(scores, 2);  % take average scores over 10 crops

[~, maxlabel] = max(scores);

% call caffe.reset_all() to reset caffe
%caffe.reset_all();

classes = textread(caffe_root + 'examples/ilsvrc12/synset_words.txt', '%s', 'delimiter','\n');
classes(maxlabel)


% ACTUAL FEATURE EXAMINATION
%data_index = net.name2blob_index('data');
%data_blob = net.blob_vec(data_index);
%data_data = data_blob.get_data();
%size(data_data)