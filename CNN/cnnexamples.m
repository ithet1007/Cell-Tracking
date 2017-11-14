clear all; close all; clc;
addpath('../data');
addpath('../util');
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


cnn = cnnsetup(cnn, train_x, train_y);   %initialize the cnn parameters(w,b)


opts.alpha = 1;

opts.batchsize = 40; 

opts.numepochs = 10;


cnn = cnntrain(cnn, train_x, train_y, opts);


[er, bad] = cnntest(cnn, test_x, test_y);


plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
