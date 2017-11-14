function trackM = initMTT(cnn, tmpl, L)

    [x1, x2] = size(tmpl.mean);
    x3 = size(tmpl.basis, 2);
    test_y = zeros(2, x3);  
    for i = 1:x3
        test_x(:,:,i) = reshape(tmpl.basis(:,i), x1, x2);
        if L(i)==1
            test_y(1,i) = 1;
        else
            test_y(2,i) = 1;
        end    
    end
    
    trackM = cnn;
    trackM.cnn = cnn;
    onum = 2;
    fvnum = size(trackM.ffW, 2);
    trackM.ffb = zeros(onum, 1);
    trackM.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));

    trackM.lambda = 1e-4;
    trackM.activation_function = 'sigm';
    opts.alpha = 1e-1;
    opts.numepochs = 2;
    opts.batchsize = 10;
    trackM.task = 'main';
    
    L(L == -1) = 0;
 
    trackM = cnntrain(trackM, test_x, test_y, opts);
end