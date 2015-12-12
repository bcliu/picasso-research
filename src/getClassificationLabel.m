function [strClass, net] = getClassificationLabel(imagePath)
    caffe.set_mode_cpu();
    model_dir = 'C:\Users\chenliu\SandBox\bvlc_alexnet\';
    net_model = [model_dir 'deploy.prototxt'];
    net_weights = [model_dir 'bvlc_alexnet.caffemodel'];
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    net = caffe.Net(net_model, net_weights, phase);
    im = imread(imagePath);

    % prepare oversampled input
    % input_data is Height x Width x Channel x Num
    input_data = {prepare_image(im)};

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

    classes = textread('W:\caffe\examples\ilsvrc12\synset_words.txt', '%s', 'delimiter','\n');
    strClass = classes(maxlabel);