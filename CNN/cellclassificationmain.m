clear all; close all; clc;
addpath('../util');

load offline_data.mat
%% ex1 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 10, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 30, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

cnn.optfun = 'softmax';
cnn.task = 'assist';
%cnn.optfun = 'sigm';
cnn.lambda = 0.0001;

cnn = cnnsetup(cnn, train_x, train_y);   %initialize the cnn parameters(w,b)

opts.alpha = 1;

opts.batchsize = 901; 

opts.numepochs = 20;

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    cnn = cnntrain(cnn, train_x, train_y, opts);
    [er, bad] = cnntest(cnn, test_x, test_y);
    %show test error
    disp([num2str(er*100) '% error']);
end
save('cnn_model.mat','cnn');

