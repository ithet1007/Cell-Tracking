function tmpl = samplePos_DLT(frame, param0, sz)
    c = 1;
    geom = affparam2geom(param0);
    h = round(sz(2)*geom(3));
    w = round(sz(1)*geom(3)*geom(5));
    for i = -1 : 1
        for j = -1 : 1
            temp = warpimg(frame, param0 + [i  j 0 0 0 0], sz);
            tmpl(:, c) = temp(:);

            c = c + 1;
        end
    end
    temp = warpimg(frame, param0, sz);
    tmpl(:, c) = temp(:);

end