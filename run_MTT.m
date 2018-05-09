% Copyright (C) Tao He, Hua Mao, and Zhang Yi.
% All rights reserved.
% The code is based on NaiYan Wang, thanks to him for sharing his code
% The CNN source code comes from matlab toolbox
% single object cell tracking
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you reuse our codes or our dataset, please cite our paper: url = "http://www.sciencedirect.com/science/article/pii/S0262885616302001",
% @article{HE2017142,
% title = "Cell tracking using deep neural networks with multi-task learning",
% journal = "Image and Vision Computing",
% volume = "60",
% pages = "142 - 153",
% year = "2017",
% note = "Regularization Techniques for High-Dimensional Data Analysis",
% issn = "0262-8856",
% doi = "https://doi.org/10.1016/j.imavis.2016.11.010",
% author = "Tao He and Hua Mao and Jixiang Guo and Zhang Yi",
% }
% download dataset please go to http://legacy.machineilab.org/users/hetao/

clear all
dataPath = '.\samples\';
sample_num = 80;
load '.\samples\groundtruth.mat';

for index = 1:sample_num
    title = num2str(index);
    pos = results{1, index}(:,1);
    p1 = (pos(1)+pos(3))/2;
    p2 = (pos(2)+pos(4))/2;
    p3 = pos(3)-pos(1);
    p4 = pos(4)-pos(2);
    p = [p1 p2 p3 p4 0];
    opt = struct('numsample',1000, 'affsig',[4,4,.005,.000,.001,.000]);
    opt.maxbasis = 10;
    opt.updateThres = 0.8;
    opt.condenssig = 0.01;
    opt.tmplsize = [32, 32];
    opt.normalWidth = 320;
    opt.normalHeight = 240;
    seq.init_rect = [p(1) - p(3) / 2, p(2) - p(4) / 2, p(3), p(4), p(5)];
    % Load data
    disp('Loading data...');
    fullPath = [dataPath, title, '\'];
    d = dir([fullPath, '*.jpg']);
    if size(d, 1) == 0
        d = dir([fullPath, '*.png']);
    end
    if size(d, 1) == 0
        d = dir([fullPath, '*.bmp']);
    end
    im = imread([fullPath, d(1).name]);
    data = zeros(size(im, 1), size(im, 2), size(d, 1));
    seq.s_frames = cell(size(d, 1), 1);
    for i = 1 : size(d, 1)
        seq.s_frames{i} = [fullPath, d(i).name];
    end
    seq.opt = opt;
    predicts = run_tracker(seq, '', false);

end
