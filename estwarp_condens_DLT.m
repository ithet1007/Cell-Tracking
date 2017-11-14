function param = estwarp_condens_DLT(frm, tmpl, param, opt, mtt, frameNum)
n = opt.numsample;
sz = size(tmpl.mean);
N = sz(1)*sz(2);

if ~isfield(param,'param')
  param.param = repmat(affparam2geom(param.est(:)), [1,n]);
else
  cumconf = cumsum(param.conf);
  idx = floor(sum(repmat(rand(1,n),[n,1]) > repmat(gather(cumconf),[1,n])))+1;
  param.param = param.param(:,idx);
end
param.param = param.param + randn(6,n).*repmat(opt.affsig(:),[1,n]);
wimgs = warpimg(frm, affparam2mat(param.param), sz);
data = reshape(wimgs,[N,n]);
mtt.task = 'assist';
mtt.cellresult = cnncelltest(mtt.cnn, wimgs);
mtt.task = 'main';
mtt = cnnff(mtt, wimgs);
confidence = mtt.o(1,:);

disp(max(confidence));
if max(confidence) < opt.updateThres || frameNum - param.lastUpdate >= 50
    param.update = true;
    param.lastUpdate = frameNum;
else
    param.update = false;
end
confidence = confidence - min(confidence);
param.conf = exp(double(confidence) ./opt.condenssig)';
param.conf = param.conf ./ sum(param.conf);
[maxprob,maxidx] = max(param.conf);
if maxprob == 0 || isnan(maxprob)
    error('overflow!');
end
param.maxprob = maxprob;
param.est = affparam2mat(param.param(:,maxidx));
param.wimg = reshape(data(:,maxidx), sz);

if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end