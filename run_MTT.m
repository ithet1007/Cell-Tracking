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
dataPath = '.\samples\'; % the dataset
sample_num = 1; %total 80 samples;
load '.\samples\groundtruth.mat';
threshold_ce = [10 15 20 25 30 35 40];
%center_errors = zeros(size(threshold_ce,2),sample_num);
center_location = zeros(1,sample_num);
threshold_sp = [0.2 0.3 0.4 0.5 0.6 0.7 0.8];
success_plots = zeros(size(threshold_sp,2), sample_num);

for index = 1:sample_num
    title = num2str(index);
    pos = results{1, index}(:,1);
    p1 = (pos(1)+pos(3))/2;
    p2 = (pos(2)+pos(4))/2;
    p3 = pos(3)-pos(1);
    p4 = pos(4)-pos(2);
    p = [p1 p2 p3 p4 0];
    opt = struct('numsample',1000, 'affsig',[4,4,.005,.000,.001,.000]);
    opt.maxbasis = 20;
    opt.base_conf = zeros(1,20);
    for i=1:opt.maxbasis 
       opt.base_conf(:,i) = (opt.maxbasis-i+1)/(opt.maxbasis-i+1+0.25); 
    end
    opt.updateThres = 0.2;
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
    center_location(:,index) = get_location(results{1,index}, predicts.res, threshold_ce);
    %center_location(:,index) = get_precision_plot(results{1,index}, predicts.res, threshold_ce);
    %success_plots(:,index) = get_success_plot(results{1,index}, predicts.res, threshold_sp);
end
save('./results/MTT0319_center_location_MLM.mat','center_location');
%save('MTT0319_success_errors_single_object_MLM.mat','success_plots');


