% Copyright (C) Tao He, Hua Mao, and Zhang Yi.
% All rights reserved.


function [ pos, conf ] = pos_queue( pos, conf, cur_conf, cur_pos )
leng = size(conf,2);
threshold = 0.1;
min = 1;
min_index = 0;
for j=1:leng
   if min > conf(:,j)
      min = conf(:,j);
      min_index = j;
   end
end

if cur_conf>min
    
    if min_index>1
       for i=min_index:-1:2
           pos(:,i) = pos(:,i-1);
           conf(:,i) = conf(:,i-1);
       end     
    end
    pos(:,1) = cur_pos;
    conf(:,1) = cur_conf;
end


for i=1:leng
   conf(:,i) = conf(:,i)*(leng-i+1)/(leng-i+1+threshold); 
end

end

