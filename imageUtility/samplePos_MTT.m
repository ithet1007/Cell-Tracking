function tmpl = samplePos_MTT(frame, param0, sz, index)
    mean = 0;
    vary = 0.00001;
    geom = affparam2geom(param0);
    h = round(sz(2)*geom(3));
    w = round(sz(1)*geom(3)*geom(5));
    temp = warpimg(frame, param0, sz);
    tmpl = imnoise(temp(:), 'gaussian',mean, vary*(index-1));
end