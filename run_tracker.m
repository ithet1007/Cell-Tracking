% Copyright (C) Tao He, Hua Mao, and Zhang Yi.
%% Thanks Naiyan Wang and Dit-Yan Yeung.
%% The related work refers to Learning A Deep Compact Image Representation for Visual Tracking. (NIPS2013')
%% All rights reserved.

load cnn_model.mat  % the pre_trained offline model could download from http://legacy.machineilab.org/users/hetao/

function results=run_tracker(seq, res_path, bSaveImage)
    addpath('affineUtility');
    addpath('drawUtility');
    addpath('imageUtility');
    addpath('CNN');
    addpath('util');
    rand('state',0);  randn('state',0);
    
    if isfield(seq, 'opt')
        opt = seq.opt;
    else
        trackparam_MTT;
    end
    rect=seq.init_rect;
    p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4), 0];
    frame = imread(seq.s_frames{1});
    if size(frame,3)==3
        frame = double(rgb2gray(frame));
    end
    
    scaleHeight = size(frame, 1) / opt.normalHeight;
    scaleWidth = size(frame, 2) / opt.normalWidth;
    p(1) = p(1) / scaleWidth;
    p(3) = p(3) / scaleWidth;
    p(2) = p(2) / scaleHeight;
    p(4) = p(4) / scaleHeight;
    frame = imresize(frame, [opt.normalHeight, opt.normalWidth]);
    frame = double(frame) / 255;
    
    paramOld = [p(1), p(2), p(3)/opt.tmplsize(2), p(5), p(4) /p(3) / (opt.tmplsize(1) / opt.tmplsize(2)), 0];
    param0 = affparam2mat(paramOld);
    
    if ~exist('opt','var')  opt = [];  end
    if ~isfield(opt,'minopt')
      opt.minopt = optimset; opt.minopt.MaxIter = 25; opt.minopt.Display='off';
    end
    reportRes = [];
    tmpl.mean = warpimg(frame, param0, opt.tmplsize);
    tmpl.basis = [];
    % Sample 10 positive templates for initialization
    %for i = 1 : opt.maxbasis / 10
    for i = 1 : opt.maxbasis
        %tmpl.basis( :, (i - 1) * 10 + 1 : i * 10) = samplePos_DLT(frame, param0, opt.tmplsize);
        tmpl.basis( :, i) = samplePos_MTT(frame, param0, opt.tmplsize, i);
    end
    % Sample 100 negative templates for initialization
    p0 = paramOld(5);
    tmpl.basis( :, opt.maxbasis + 1 : 100 + opt.maxbasis) = sampleNeg(frame, param0, opt.tmplsize, 100, opt, 8);

    param.est = param0;
    param.lastUpdate = 1;

    wimgs = [];

    % draw initial track window
    drawopt = drawtrackresult([], 0, frame, tmpl, param, []);
    drawopt.showcondens = 0;  drawopt.thcondens = 1/opt.numsample;
    if (bSaveImage)
        imwrite(frame2im(getframe(gcf)),sprintf('%s0000.jpg',res_path));    
    end
    
    % track the sequence from frame 2 onward
    duration = 0; 
    if (exist('dispstr','var'))  dispstr='';  end
    L = [ones(opt.maxbasis, 1); (-1) * ones(100, 1)];
    mtt = initMTT(cnn, tmpl, L);
    L = [];
    tic;
    pos = tmpl.basis(:, 1 : opt.maxbasis);
    opts.numepochs = 5 ;
    for f = 1:size(seq.s_frames,1)  
      frame = imread(seq.s_frames{f});
      if size(frame,3)==3
        frame = double(rgb2gray(frame));
      end  
      frame = imresize(frame, [opt.normalHeight, opt.normalWidth]);
      frame = double(frame) / 255;

      % do tracking
       param = estwarp_condens_DLT(frame, tmpl, param, opt, mtt, f);

      % do update

      temp = warpimg(frame, param.est', opt.tmplsize);
      
      % process the queue
      [pos, opt.base_conf] = pos_queue(pos,opt.base_conf, param.maxprob, temp(:));
      %pos(:, mod(f - 1, opt.maxbasis) + 1) = temp(:);
      if  param.update
          opts.batchsize = 10;
          opts.alpha = 1e-1;
          % Sample two set of negative samples at different range.
          neg = sampleNeg(frame, param.est', opt.tmplsize, 50, opt, 8);
          neg = [neg sampleNeg(frame, param.est', opt.tmplsize, 50, opt, 4)];
            samples = [pos neg];
            [x1, x2] = size(tmpl.mean);
            x3 = size(samples, 2);
            train_y = zeros(2, x3);  
            for i = 1:x3
                train_x(:,:,i) = reshape(samples(:,i), x1, x2);
            end
            train_y(1,:) = [ones(opt.maxbasis , 1); zeros(100, 1)];
            train_y(2,:) = [zeros(opt.maxbasis, 1); ones(100, 1)];
          mtt = cnntrain(mtt, train_x, train_y, opts);
      end

      duration = duration + toc;
      
      res = affparam2geom(param.est);
      p(1) = round(res(1));
      p(2) = round(res(2)); 
      p(3) = round(res(3) * opt.tmplsize(2));
      p(4) = round(res(5) * (opt.tmplsize(1) / opt.tmplsize(2)) * p(3));
      p(5) = res(4);
      p(1) = p(1) * scaleWidth;
      p(3) = p(3) * scaleWidth;
      p(2) = p(2) * scaleHeight;
      p(4) = p(4) * scaleHeight;
      paramOld = [p(1), p(2), p(3)/opt.tmplsize(2), p(5), p(4) /p(3) / (opt.tmplsize(1) / opt.tmplsize(2)), 0];
      reportRes = [reportRes;  affparam2mat(paramOld)];
      
      tmpl.basis = [pos];
      drawopt = drawtrackresult(drawopt, f, frame, tmpl, param, []);
      if (bSaveImage)
          imwrite(frame2im(getframe(gcf)),sprintf('%s/%04d.jpg',res_path,f));
      end
      tic;
    end
    duration = duration + toc
    fprintf('%d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
    results.res=reportRes;
    results.type='ivtAff';
    results.tmplsize = opt.tmplsize;
    results.fps = f/duration;
end
